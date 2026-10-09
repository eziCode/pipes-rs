import csv
import tempfile
import unittest
from pathlib import Path
from run_execution_pilot import inspect

class ExecutionContract(unittest.TestCase):
    def check_rows(self, rows, capacity=1):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'raw.csv'
            with path.open('w') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
            return inspect(path,len(rows),capacity)
    @staticmethod
    def row(i,start):
        return dict(index=i,timestamp=100+i,points=10,boxes=1,started_ns=start,received_ns=start+2,
                    finished_ns=start+10,handoff_ns=2,fusion_ns=8,result_ns=10)
    def test_valid_sequential(self):
        self.assertEqual(len(self.check_rows([self.row(0,10),self.row(1,25)])),2)
    def test_excess_outstanding_work(self):
        with self.assertRaisesRegex(ValueError,'outstanding'):
            self.check_rows([self.row(0,10),self.row(1,11)])
    def test_clock_mismatch(self):
        row=self.row(0,10);row['fusion_ns']=7
        with self.assertRaisesRegex(ValueError,'clock'):self.check_rows([row])
    def test_duplicate_sample(self):
        row=self.row(1,25);row['timestamp']=100
        with self.assertRaisesRegex(ValueError,'duplicate'):self.check_rows([self.row(0,10),row])

if __name__=='__main__':unittest.main()
