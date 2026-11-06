import unittest
from dictophone.core import Turn
from dictophone.refinement import refine_labels


class ShortTurnRefinementTests(unittest.TestCase):
    def test_short_turn_moves_to_matching_voice_without_moving_time(self):
        turns=[Turn(0,5000,'a'),Turn(6000,11000,'b'),Turn(12000,13000,'b')]
        result,count=refine_labels(turns,[[1,0],[0,1],[1,0]])
        self.assertEqual(count,1)
        self.assertEqual(result[-1],Turn(12000,13000,'a'))
        self.assertEqual(turns[-1].speaker,'b')

    def test_ambiguous_embedding_does_not_force_label(self):
        turns=[Turn(0,5000,'a'),Turn(6000,11000,'b'),Turn(12000,13000,'b')]
        result,count=refine_labels(turns,[[1,0],[0.99,0.01],[1,0]])
        self.assertEqual(count,0)
        self.assertEqual(result,turns)

    def test_target_is_not_its_own_reference(self):
        turns=[Turn(0,2000,'a'),Turn(3000,5000,'b')]
        result,count=refine_labels(turns,[[1,0],[1,0]])
        self.assertEqual(count,0)

    def test_overlapping_speech_is_never_reassigned(self):
        turns=[Turn(0,5000,'a'),Turn(6000,11000,'b'),Turn(12000,14000,'a'),Turn(12500,13500,'b')]
        result,count=refine_labels(turns,[[1,0],[0,1],[0,1],[1,0]])
        self.assertEqual(result[-2:],turns[-2:])
        self.assertEqual(count,0)
