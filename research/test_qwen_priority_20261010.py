import unittest
from qwen_priority_compile_20261010 import earliest_invalid


class Decisions(unittest.TestCase):
    def test_alternative_order_and_shared_delimiter(self):
        d=earliest_invalid([10,20,30,80,99],[[10,20,30,40,99],[10,20,50,60,99]],1,{99})
        self.assertEqual(d['context_ids'],[10,20,30])
        self.assertEqual(d['allowed_ids'],[40])
        self.assertEqual(d['rejected_id'],80)

    def test_terminal_punctuation(self):
        d=earliest_invalid([10,20,70,99],[[10,99],[10,20,99]],1,{98,99})
        self.assertEqual(d['context_ids'],[10,20])
        self.assertEqual(d['allowed_ids'],[98,99])

    def test_legal_rejection_refused(self):
        with self.assertRaises(ValueError):earliest_invalid([10,99],[[10,99]],1,{99})

    def test_no_shared_context(self):
        with self.assertRaises(ValueError):earliest_invalid([10,80,99],[[11,40,99]],1,{99})


if __name__=='__main__':unittest.main()
