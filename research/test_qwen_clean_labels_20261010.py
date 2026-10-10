import unittest
from qwen_clean_labels_20261010 import repair,complete
from completion_pairs_20261010 import covered,true_repeat


def edge(name,doc):return {'answer':name,'aliases':[name],'doc_id':doc}


class CleanLabels(unittest.TestCase):
    def test_duplicate_gold_group_single_unit(self):
        edges=[edge('Alpha',1),edge('Beta',2),edge('Beta',2)]
        text=complete('Alpha [1]',edges)
        self.assertEqual(text,'Alpha [1], Beta [2]')
        self.assertEqual(len(covered(text,edges)),3)

    def test_wrong_old_citation_does_not_repeat_entity(self):
        row={'trajectory_prefix':'Alpha [2]','trajectory_continuation':', Alpha [2]',
             'edges':[edge('Alpha',1),edge('Beta',2)]}
        s=repair(row)
        self.assertEqual(s['prefix'],'Alpha [1]')
        self.assertEqual(s['chosen'],'Alpha [1], Beta [2]')
        self.assertFalse(true_repeat({'short':s['chosen']}))

    def test_same_entity_two_source_citation_targets(self):
        text=complete('Alpha [1]',[edge('Alpha',1),edge('Alpha',2)])
        self.assertEqual(text,'Alpha [1][2]')

    def test_complete_after_citation_correction_stops(self):
        row={'trajectory_prefix':'Alpha [2]','trajectory_continuation':', Alpha [2]','edges':[edge('Alpha',1)]}
        self.assertEqual(repair(row)['action'],'stop')


if __name__=='__main__':unittest.main()
