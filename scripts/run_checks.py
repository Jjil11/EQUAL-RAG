"""CPU tests using synthetic evidence, never undistributed benchmark rows."""
import sys,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research'),str(ROOT/'tests')]

def main():
    import test_strict_units_20261010 as metrics
    import test_completion_repair_20261010 as repair
    import test_release as publication
    import test_supplementary_release as supplementary
    suite=unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(metrics))
    # Explicitly not running the historical test that reads real prepared rows.
    for name in unittest.defaultTestLoader.getTestCaseNames(repair.SafetyTests):
        if name!='test_all_existing_training_labels':suite.addTest(repair.SafetyTests(name))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(publication))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(supplementary))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print('Scope: synthetic metric/compiler + release checks; no GPU training or real-data judge validation.')
    raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__=='__main__':main()
