import sys
sys.path.insert(0, '../')

import unittest

from collections import OrderedDict
from map_parser import MapParser


class TestMapParser(unittest.TestCase):

    def setUp(self):
        self.s1 = MapParser('../examples/Niko - Made of Fire (lesjuh) [Oni].osu')
        self.s2 = MapParser("../examples/TERRASPEX - AMAZING BREAK (Monstrata) [APPLE'S EXPERT].osu")

    def test_version(self):
        self.assertEqual(self.s1.version, '6')
        self.assertEqual(self.s2.version, '14')

    def test_general(self):
        expected_result = {
            'AudioFilename': 'Niko - Made of Fire.mp3',
            'AudioLeadIn': '1000',
            'PreviewTime': '40703',
            'Countdown': '1',
            'SampleSet': 'Normal',
            'StackLeniency': '0.7',
            'Mode': '0',
            'LetterboxInBreaks': '1'
        }
        self.assertEqual(self.s1.general, expected_result)
        expected_result = {
            'AudioFilename': 'audio.mp3',
            'AudioLeadIn': '0',
            'PreviewTime': '131982',
            'Countdown': '0', 'SampleSet': 'None',
            'StackLeniency': '0.3',
            'Mode': '0',
            'LetterboxInBreaks': '0',
            'WidescreenStoryboard': '0'
        }
        self.assertEqual(self.s2.general, expected_result)

    def test_editor(self):
        expected_result = {
            'DistanceSpacing': '1.70000004768372',
            'BeatDivisor': '4',
            'GridSize': '8'
        }
        self.assertEqual(self.s1.editor, expected_result)
        expected_result = {
            'Bookmarks': '7420,19041,20315,32864,34433,46982,53256,59531,64237,65805,70511,72080,76786,78354,84629,90903,97178,101884,103452,106590,112864,114237,119139,125413,128550,132472,137177,138746,143452,145021,150511,151295,154432,157570,161491,166196,167765,173451,175608,177177,180314,183451,185020,186589,188157,190118,190511,195216,196785,201491,203060,209334,214040,215609,221883',
            'DistanceSpacing': '1.3',
            'BeatDivisor': '4',
            'GridSize': '4',
            'TimelineZoom': '3.899998'
        }
        self.assertEqual(self.s2.editor, expected_result)

    def test_metadata(self):
        expected_result = {
            'Title': 'Made of Fire',
            'Artist': 'Niko',
            'Creator': 'lesjuh',
            'Version': 'Oni',
            'Source': '',
            'Tags': 'insane stepmania'
        }
        self.assertEqual(self.s1.metadata, expected_result)
        expected_result = {
            'Title': 'AMAZING BREAK',
            'TitleUnicode': 'AMAZING BREAK',
            'Artist': 'TERRASPEX',
            'ArtistUnicode': 'TERRASPEX',
            'Creator': 'Monstrata',
            'Version': "APPLE'S EXPERT",
            'Source': 'TERRA FORMARS',
            'Tags': 'テラフォーマーズ appleeaterx hobbes2 deppyforce derandom_otaku',
            'BeatmapID': '1217004',
            'BeatmapSetID': '571835'
        }
        self.assertEqual(self.s2.metadata, expected_result)

    def test_difficulty(self):
        expected_result = {
            'HPDrainRate': '6',
            'CircleSize': '4',
            'OverallDifficulty': '8',
            'SliderMultiplier': '2',
            'SliderTickRate': '1'
        }
        self.assertEqual(self.s1.difficulty, expected_result)
        expected_result = {
            'HPDrainRate': '6',
            'CircleSize': '3.5',
            'OverallDifficulty': '8.5',
            'ApproachRate': '9.3',
            'SliderMultiplier': '3.1',
            'SliderTickRate': '1'
        }
        self.assertEqual(self.s2.difficulty, expected_result)

    def test_events(self):
        expected_result = OrderedDict([
            ('0', ['0', '"maid-bg.png"']),
            ('3', ['100', '0', '0', '0'])
        ])
        self.assertEqual(self.s1.events, expected_result)
        expected_result = OrderedDict([
            ('0', ['0', '"831a24cb6237bf53abc3bc12edff7a78.jpg"', '0', '0'])
        ])
        self.assertEqual(self.s2.events, expected_result)

    def test_timingpoints(self):
        expected_result = OrderedDict([
            ('368', ['368.098159509202', '4', '1', '0', '80', '1', '0']),
            ('3678', ['370.37037037037', '4', '1', '0', '80', '1', '0']),
            ('15518', ['370.37037037037', '4', '1', '0', '80', '1', '0']),
            ('40718', ['370.37037037037', '4', '1', '0', '80', '1', '0']),
            ('64398', ['370.37037037037', '4', '1', '0', '80', '1', '0'])
        ])
        self.assertEqual(self.s1.timingpoints, expected_result)
        first_obj_key = next(iter(self.s2.timingpoints))
        expected_result = ['-200', '4', '3', '0', '70', '0', '0']
        self.assertEqual(self.s2.timingpoints[first_obj_key], expected_result)
        last_obj_key = next(reversed(self.s2.timingpoints))
        expected_result = ['-86.9565217391304', '4', '3', '1', '70', '0', '0']
        self.assertEqual(self.s2.timingpoints[last_obj_key], expected_result)

    def test_colours(self):
        expected_result = {
            'Combo1': ('159', '159', '0'),
            'Combo2': ('113', '0', '0'),
            'Combo3': ('145', '72', '0')
        }
        self.assertEqual(self.s1.colours, expected_result)
        expected_result = {
            'Combo1': ('170', '87', '87'),
            'Combo2': ('164', '11', '11'),
            'Combo3': ('64', '128', '128'),
            'Combo4': ('198', '145', '196')
        }
        self.assertEqual(self.s2.colours, expected_result)

    def test_hitobjects(self):
        first_obj_key = next(iter(self.s1.hitobjects))
        expected_result = '256,72,368,1,4'
        self.assertEqual(self.s1.hitobjects[0], expected_result)
        last_obj_key = next(reversed(self.s1.hitobjects))
        expected_result = '256,48,77731,5,4'
        self.assertEqual(self.s1.hitobjects[-1], expected_result)
        first_obj_key = next(iter(self.s2.hitobjects))
        expected_result = '13,138,800,6,0,B|139:96|99:181|238:140,1,232.5,4|0,0:1|0:0,0:0:0:0:'
        self.assertEqual(self.s2.hitobjects[0], expected_result)
        last_obj_key = next(reversed(self.s2.hitobjects))
        expected_result = '168,132,229627,1,2,1:3:0:0:'
        self.assertEqual(self.s2.hitobjects[-1], expected_result)
