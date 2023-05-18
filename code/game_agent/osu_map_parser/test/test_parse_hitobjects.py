import sys
sys.path.insert(0, '../')

import unittest

from hitobject_parser import HitObjectParser
from hitobject_type import (Position, Hitsounds,
                            SlideHitsounds, SlidePosition, SlideVoiceEffect,
                            SpinTimestamp,
                            Circle, Slide, Spin)


class TestHitobjectParser(unittest.TestCase):

    def test_empty_line(self):
        self.assertEqual(None, HitObjectParser.parse(''))

    def test_incomplete_line(self):
        self.assertEqual(None, HitObjectParser.parse('456,206,1903,'))

    def test_circle_without_hitsounds(self):
        expected_result = Circle(
            pos=Position(x=143, y=139),
            timestamp=24923,
            is_start=False,
            voice_effect=0,
            hitsounds=Hitsounds(sampleset=0, additions=0)
        )
        self.assertEqual(expected_result, HitObjectParser.parse('143,139,24923,1,0'))

    def test_circle_with_hitsounds(self):
        expected_result = Circle(
            pos=Position(x=143, y=139),
            timestamp=24923,
            is_start=False,
            voice_effect=0,
            hitsounds=Hitsounds(sampleset=1, additions=2)
        )
        self.assertEqual(expected_result, HitObjectParser.parse('143,139,24923,1,0,1:2:0:0:'))

    def test_slide_without_hitsounds(self):
        expected_result = Slide(
            pos=SlidePosition(
                start=Position(x=472, y=256),
                end=Position(x=447, y=300),
                body=None
            ),
            timestamp=74583,
            slide_type='B',
            is_start=False,
            tick=50.0,
            voice_effect=SlideVoiceEffect(start=0, body=0, end=0),
            hitsounds=SlideHitsounds(
                start=Hitsounds(sampleset=0, additions=0),
                body=Hitsounds(sampleset=0, additions=0),
                end=Hitsounds(sampleset=0, additions=0)
            )
        )
        self.assertEqual(expected_result, HitObjectParser.parse('472,256,74583,2,0,B|447:300,1,50,0|0'))

    def test_slide_without_hitsounds_and_vocal_effects(self):
        expected_result = Slide(
            pos=SlidePosition(
                start=Position(x=104, y=256),
                end=Position(x=112, y=40),
                body=None
            ),
            timestamp=63680,
            slide_type='B',
            is_start=True,
            tick=200.0,
            voice_effect=SlideVoiceEffect(start=0, body=6, end=0),
            hitsounds=SlideHitsounds(
                start=Hitsounds(sampleset=0, additions=0),
                body=Hitsounds(sampleset=0, additions=0),
                end=Hitsounds(sampleset=0, additions=0)
            )
        )
        self.assertEqual(expected_result, HitObjectParser.parse('104,256,63680,6,6,B|112:40,1,200'))

    def test_slide_in_curve_shape(self):
        expected_result = Slide(
            pos=SlidePosition(
                start=Position(x=416, y=315),
                end=Position(x=274, y=259),
                body=[Position(x=323, y=302), Position(x=375, y=261)]
            ),
            timestamp=26982,
            slide_type='B',
            is_start=False,
            tick=155.0,
            voice_effect=SlideVoiceEffect(start=0, body=0, end=0),
            hitsounds=SlideHitsounds(
                start=Hitsounds(sampleset=0, additions=0),
                body=Hitsounds(sampleset=0, additions=0),
                end=Hitsounds(sampleset=0, additions=0)
            )
        )
        self.assertEqual(expected_result, HitObjectParser.parse('416,315,26982,2,0,B|323:302|375:261|274:259,1,155,10|0'))

    def test_slide_with_hitsounds(self):
        expected_result = Slide(
            pos=SlidePosition(
                start=Position(x=0, y=192),
                end=Position(x=368, y=192),
                body=None
            ),
            timestamp=350,
            slide_type='L',
            is_start=False,
            tick=360.0,
            voice_effect=SlideVoiceEffect(start=2, body=8, end=4),
            hitsounds=SlideHitsounds(
                start=Hitsounds(sampleset=1, additions=2),
                body=Hitsounds(sampleset=3, additions=1),
                end=Hitsounds(sampleset=2, additions=3)
            )
        )
        self.assertEqual(expected_result, HitObjectParser.parse('0,192,350,2,8,L|368:192,1,360,2|4,1:2|3:1,2:3:0:0:'))

    def test_spin_with_hitsounds(self):
        expected_result = Spin(
            timestamp=SpinTimestamp(start=2800, end=3250),
            voice_effect=14,
            hitsounds=Hitsounds(sampleset=1, additions=2)
        )
        self.assertEqual(expected_result, HitObjectParser.parse('256,192,2800,12,14,3250,1:2:0:0:'))

    def test_spin_without_hitsounds(self):
        expected_result = Spin(
            timestamp=SpinTimestamp(start=2800, end=3250),
            voice_effect=14,
            hitsounds=Hitsounds(sampleset=0, additions=0)
        )
        self.assertEqual(expected_result, HitObjectParser.parse('256,192,2800,12,14,3250'))
