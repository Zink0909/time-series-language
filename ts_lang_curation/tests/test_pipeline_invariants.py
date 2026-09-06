import math
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock


PKG = Path(__file__).resolve().parents[1]
COMPANION = PKG / "flywheel" / "companion"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(COMPANION))

from core import chatml  # noqa: E402
from core.emitters import emit_cpt, emit_ir  # noqa: E402
from core.governance import apply_governance, release_errors  # noqa: E402
from core.ir import from_record  # noqa: E402
from core.schema import TextToTs, TsToText  # noqa: E402
from sources.official_series import parse_treasury_yield_xml  # noqa: E402
from core.writer import write_jsonl  # noqa: E402
import arms  # noqa: E402
import data  # noqa: E402
import migas_finetune  # noqa: E402
import verify_any  # noqa: E402
import build as unified_build  # noqa: E402
import release as release_catalog  # noqa: E402


def _items(n=20):
    return [{"series_id": f"s{i}", "series_group": f"g{i}", "text": f"text-{i}"}
            for i in range(n)]


class PipelineInvariantTests(unittest.TestCase):
    def test_shuffled_arm_is_a_derangement_and_preserves_text_multiset(self):
        for n in (2, 3, 40, 2000):
            with self.subTest(n=n):
                result = arms.make_arms(_items(n), seed=7)
                b = result["B_flywheel_text"]
                c = result["C_shuffled_text"]
                self.assertEqual(Counter(x["text"] for x in b), Counter(x["text"] for x in c))
                self.assertTrue(all(left["text"] != right["text"] for left, right in zip(b, c)))

    def test_shuffled_arm_fails_closed_for_single_item(self):
        with self.assertRaises(ValueError):
            arms.make_arms(_items(1))

    def test_shuffled_arm_preserves_dataset_strata(self):
        source = []
        for dataset in ("a", "b"):
            source.extend({"series_id": f"{dataset}-{i}", "dataset": dataset,
                           "text": f"{dataset}-text-{i}"} for i in range(3))
        shuffled = arms.make_arms(source, seed=3)["C_shuffled_text"]
        self.assertTrue(all(row["text"].startswith(row["dataset"] + "-") for row in shuffled))

    def test_shuffleable_partition_makes_singleton_exclusion_explicit(self):
        source = [
            {"dataset": "a", "series_id": "a1"},
            {"dataset": "a", "series_id": "a2"},
            {"dataset": "b", "series_id": "b1"},
        ]
        kept, dropped = arms.partition_shuffleable(source)
        self.assertEqual([x["series_id"] for x in kept], ["a1", "a2"])
        self.assertEqual([x["series_id"] for x in dropped], ["b1"])

    def test_time_split_can_purge_groups_and_checks_record_knowledge_time(self):
        items = [
            {"series_id": "a-old", "series_group": "a", "origin": "2023-01-02",
             "kt": "2023-01-01T00:00:00Z"},
            {"series_id": "a-new", "series_group": "a", "origin": "2024-01-02",
             "kt": "2024-01-01T00:00:00Z"},
            {"series_id": "b-old", "series_group": "b", "origin": "2023-01-02",
             "kt": "2023-01-01T00:00:00Z"},
        ]
        train, test, problems = data.time_split(items, "2024-01-01", group_disjoint=True)
        self.assertEqual([x["series_id"] for x in train], ["b-old"])
        self.assertEqual([x["series_id"] for x in test], ["a-new"])
        self.assertEqual(problems, [])

    def test_chatml_rejects_invalid_numeric_values_and_empty_future(self):
        with self.assertRaises(ValueError):
            chatml._z([1.0, math.inf])
        with self.assertRaises(ValueError):
            chatml.build_ts_forecast("event", [1.0, 2.0], [], "x", "u", "daily", {})

    def test_writer_replaces_artifact_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            self.assertEqual(write_jsonl([{"n": 1}, {"n": 2}], path), 2)
            self.assertEqual(path.read_text(encoding="utf-8"), '{"n": 1}\n{"n": 2}\n')
            self.assertEqual(list(Path(directory).glob(".jsonl-*.tmp")), [])

    def test_unknown_format_is_not_assumed_to_be_cpt(self):
        self.assertEqual(verify_any.detect_format({"text": "untyped", "timeseries": []}), "unknown")

    def test_validators_return_errors_instead_of_crashing_on_malformed_spans(self):
        malformed = {"text": "<|im_start|><ts></ts>", "task_type": "ts_forecast",
                     "timeseries": ["not-an-object"], "normalization": {"num_spans": 1}}
        self.assertTrue(chatml.validate(malformed))
        errors, _ = verify_any.check_chatml(malformed, 2)
        self.assertTrue(errors)

    def test_release_profile_requires_governance_fields(self):
        rec = chatml.build_ts_forecast(
            "Known before the event", [1.0, 2.0], [3.0], "x", "u", "daily",
            {"dataset": "d", "series_id": "s", "knowledge_time": "2023-01-01"})
        errors, _ = verify_any.check_chatml(rec, 2, release=True)
        codes = {code for code, _ in errors}
        self.assertTrue({"license_missing", "source_missing", "source_url_missing"} <= codes)

    def test_release_profile_rejects_unapproved_generated_text(self):
        rec = chatml.build_text_desc(
            "Describe.", [{"name": "x", "values": [1.0, 2.0], "unit": "u", "freq": "daily"}],
            "It increased.", apply_governance({"dataset": "usgs_quakes", "series_id": "s",
                                               "source": "USGS", "text_url": "https://example.test",
                                               "is_generated_text": "derived_generated"}))
        errors, _ = verify_any.check_chatml(rec, 2, release=True)
        self.assertIn("generated_text_unapproved", {code for code, _ in errors})

    def test_governance_is_fail_closed_and_blocks_fred(self):
        unknown = apply_governance({"dataset": "not_registered"})
        self.assertEqual(unknown["license_status"], "review_required")
        self.assertTrue(release_errors(unknown))
        fred = apply_governance({"dataset": "fred_fomc"})
        self.assertEqual(fred["license_status"], "blocked")
        self.assertIn("license_not_releasable", {code for code, _ in release_errors(fred)})
        commodity = apply_governance({"dataset": "flywheel_commodity_rich"})
        self.assertEqual(commodity["license_status"], "blocked")

    def test_governance_allows_approved_and_attributed_conditional_sources(self):
        self.assertEqual(release_errors(apply_governance({"dataset": "usgs_quakes"})), [])
        self.assertEqual(release_errors(apply_governance({"dataset": "wiki_pageviews"})), [])
        wiki_alias = apply_governance({"dataset": "flywheel_wiki_big"})
        self.assertEqual(wiki_alias["governance_canonical_dataset"], "wiki_pageviews")
        self.assertEqual(release_errors(wiki_alias), [])

    def test_canonical_ir_round_trip_and_peer_cpt_emission(self):
        example = TsToText(
            "Describe it.", [{"name": "x", "values": [1.0, 2.0], "unit": "u", "freq": "daily"}],
            "The value increased.", meta={"dataset": "usgs_quakes", "series_id": "s"})
        ir_record = emit_ir(example)
        restored = from_record(ir_record)
        self.assertEqual(restored.series[0]["values"], [1.0, 2.0])
        cpt = emit_cpt(example)
        self.assertEqual(cpt["timeseries"][0]["values"], [1.0, 2.0])
        self.assertEqual(cpt["license_status"], "approved")

    def test_cpt_rejects_forecasting_example_without_lossy_coercion(self):
        example = TextToTs("event", [1.0, 2.0], [3.0], "x", "u", "daily")
        with self.assertRaises(ValueError):
            emit_cpt(example)

    def test_treasury_xml_parser_uses_first_party_maturities(self):
        xml = b'''<feed xmlns:m="urn:m" xmlns:d="urn:d"><entry><content><m:properties>
        <d:NEW_DATE>2024-01-02T00:00:00</d:NEW_DATE><d:BC_2YEAR>4.33</d:BC_2YEAR>
        <d:BC_10YEAR>3.95</d:BC_10YEAR></m:properties></content></entry></feed>'''
        self.assertEqual(parse_treasury_yield_xml(xml), [("2024-01-02", 4.33, 3.95)])

    def test_source_discovery_excludes_helper_modules(self):
        sources = unified_build.list_sources()
        self.assertIn("treasury_fomc", sources)
        self.assertIn("flywheel_wiki", sources)
        self.assertNotIn("official_series", sources)

    def test_recursive_verifier_ignores_build_ledgers(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "source.jsonl"
            rejected = Path(directory) / "source.jsonl.rejected.jsonl"
            unsupported = Path(directory) / "source.jsonl.unsupported.jsonl"
            filtered = Path(directory) / "source.jsonl.filtered.jsonl"
            for path in (artifact, rejected, unsupported, filtered):
                path.write_text("", encoding="utf-8")
            selected = verify_any.discover_files([directory])
            self.assertEqual(selected, [str(artifact)])

    def test_release_catalog_refuses_artifact_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "orphan.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing build manifest"):
                release_catalog.build_catalog(directory)

    def test_release_catalog_records_policy_empty_artifact_as_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "generated_only.jsonl"
            artifact.write_text("", encoding="utf-8")
            filtered = Path(str(artifact) + ".filtered.jsonl")
            filtered.write_text('{"series_id":"s","reason":"generated"}\n', encoding="utf-8")
            manifest = {
                "artifact_sha256": release_catalog.sha256(artifact),
                "builder": release_catalog.BUILDER,
                "records_written": 0,
                "records_filtered": 1,
                "records_rejected": 0,
                "records_unsupported": 0,
                "generated_policy": "real-only",
                "source": "generated_only",
            }
            Path(str(artifact) + ".manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            audit = {"n": 0, "format": "unknown", "errors": Counter(), "warns": Counter()}
            with mock.patch.object(release_catalog.verify_any, "audit_file", return_value=audit):
                with self.assertRaisesRegex(ValueError, "no non-empty releasable artifacts"):
                    release_catalog.build_catalog(directory)

    def test_release_catalog_refuses_missing_filtered_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "rows.jsonl"
            artifact.write_text('{}\n', encoding="utf-8")
            manifest = {
                "artifact_sha256": release_catalog.sha256(artifact),
                "builder": release_catalog.BUILDER,
                "records_written": 1,
                "records_filtered": 0,
                "records_rejected": 0,
                "records_unsupported": 0,
                "generated_policy": "real-only",
            }
            Path(str(artifact) + ".manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            audit = {"n": 1, "format": "chatml", "errors": Counter(), "warns": Counter()}
            with mock.patch.object(release_catalog.verify_any, "audit_file", return_value=audit):
                with self.assertRaisesRegex(ValueError, "filtered_ledger"):
                    release_catalog.build_catalog(directory)

    def test_bootstrap_resamples_persistent_series_as_clusters(self):
        deltas = [-1.0, -2.0, 3.0]
        events = [{"series_group": "same"}, {"series_group": "same"},
                  {"series_group": "other"}]
        observed, low, high, clusters = migas_finetune.bootstrap_median_delta(
            deltas, events, iters=100, seed=1)
        self.assertEqual(observed, -1.0)
        self.assertEqual(clusters, 2)
        self.assertLessEqual(low, observed)
        self.assertGreaterEqual(high, observed)


if __name__ == "__main__":
    unittest.main()
