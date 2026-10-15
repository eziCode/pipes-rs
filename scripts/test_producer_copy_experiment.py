"""Independent trial scheduling and run-level reporting."""

import collections
import unittest

from run_producer_copy_experiment import METRICS, VARIANTS, aggregate, schedule


class ExperimentTests(unittest.TestCase):
    def test_seeded_order_preserves_every_arm_at_every_rate_and_repeat(self):
        jobs = schedule(5, [0, 8], 123)
        self.assertEqual(jobs, schedule(5, [0, 8], 123))
        self.assertNotEqual(jobs, schedule(5, [0, 8], 124))
        counts = collections.Counter(
            (j["rate"], j["repeat"], j["variant"]) for j in jobs
        )
        expected = {
            (rate, repeat, variant): 1
            for rate in [0, 8]
            for repeat in range(1, 6)
            for variant in VARIANTS
        }
        self.assertEqual(counts, expected)

    def test_summary_uses_independent_run_medians_and_ranges(self):
        rows = []
        for rate in [0, 8]:
            for variant in VARIANTS:
                for value in [1, 3, 20]:
                    rows.append(
                        dict(
                            rate=rate, variant=variant, **dict.fromkeys(METRICS, value)
                        )
                    )
        summaries = aggregate(rows)
        self.assertEqual(len(summaries), 6)
        for s in summaries:
            self.assertEqual(s["runs"], 3)
            self.assertEqual(s["throughput_bundles_s_median"], 3)
            self.assertEqual(s["throughput_bundles_s_min"], 1)
            self.assertEqual(s["throughput_bundles_s_max"], 20)


if __name__ == "__main__":
    unittest.main()
