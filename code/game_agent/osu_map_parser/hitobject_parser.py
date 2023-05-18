import logging
import re

from typing import Tuple, List, Optional, Union
from hitobject_type import (Position, Hitsounds,
                            SlideHitsounds, SlidePosition, SlideVoiceEffect,
                            SpinTimestamp,
                            Circle, Slide, Spin)


class HitObjectParser:

    @staticmethod
    def parse(hitobject: str) -> Union[None, Circle, Slide, Spin]:
        splits = hitobject.split(',')
        try:
            hitobject_type = splits[3]
            if hitobject_type == '1':
                return HitObjectParser.parse_circle(hitobject, False)
            elif hitobject_type == '2':
                return HitObjectParser.parse_slide(hitobject, False)
            elif hitobject_type == '5':
                return HitObjectParser.parse_circle(hitobject, True)
            elif hitobject_type == '6':
                return HitObjectParser.parse_slide(hitobject, True)
            elif hitobject_type == '12':
                return HitObjectParser.parse_spin(hitobject)
            else:
                logging.warning('Invalid Line: {}'.format(hitobject))
                return None
        except IndexError:
            logging.warning('Invalid Line: {}'.format(hitobject))
            return None

    @staticmethod
    def parse_circle_spin_hitsounds(hitsounds: str) -> Optional[Hitsounds]:
        components = hitsounds.split(':')
        try:
            return Hitsounds(int(components[0]), int(components[1]))
        except IndexError:
            logging.warning('Invalid Hitsounds: {}'.format(hitsounds))
            return None

    @staticmethod
    def parse_circle(hitobject: str, is_start: bool) -> Optional[Circle]:
        components = hitobject.split(',')
        try:
            pos = Position(int(components[0]), int(components[1]))
            timestamp = int(components[2])
            is_start = is_start
            voice_effect = int(components[4])
            if len(components) == 6:
                hitsounds = HitObjectParser.parse_circle_spin_hitsounds(components[5])
                if hitsounds:
                    return Circle(pos, timestamp, is_start, voice_effect, hitsounds)
            return Circle(pos, timestamp, is_start, voice_effect)
        except IndexError:
            logging.warning('Invalid Hitobject: {}'.format(hitobject))
            return None

    @staticmethod
    def parse_slide_pos(pos: str) -> Optional[Position]:
        components = pos.split(':')
        try:
            return Position(int(components[0]), int(components[1]))
        except IndexError:
            logging.warning('Invalid Slide Position: {}'.format(pos))
            return None

    @staticmethod
    def parse_slide_body_pos(body_pos: List[str]) -> Optional[List[Position]]:
        if len(body_pos) == 0:
            return None
        else:
            return [HitObjectParser.parse_slide_pos(component) for component in body_pos]

    @staticmethod
    def parse_slide_posline(posline: str) -> Optional[Tuple[str, Optional[List[Position]], Position]]:
        try:
            components = posline.split('|')
            slide_type = components[0]
            body_pos = HitObjectParser.parse_slide_body_pos(components[1:-1])
            end_pos = HitObjectParser.parse_slide_pos(components[-1])
            return slide_type, body_pos, end_pos
        except IndexError:
            logging.warning('Invalid Slide Position Line: {}'.format(posline))
            return None

    @staticmethod
    def parse_slide_voice_effect(voice_effect: str) -> Optional[Tuple[int, int]]:
        components = voice_effect.split('|')
        try:
            return int(components[0]), int(components[1])
        except IndexError:
            logging.warning('Invalid Slide Vocal Effect: {}'.format(voice_effect))
            return None

    @staticmethod
    def parse_slide_hitsounds(hitsounds: str) -> Optional[SlideHitsounds]:
        components = re.findall(r'[\d]+', hitsounds)
        try:
            return SlideHitsounds(Hitsounds(int(components[0]), int(components[1])),
                                  Hitsounds(int(components[2]), int(components[3])),
                                  Hitsounds(int(components[4]), int(components[5])))
        except IndexError:
            logging.warning('Invalid Slide Hitsounds: {}'.format(hitsounds))
            return None

    @staticmethod
    def parse_slide(hitobject: str, is_start: bool) -> Optional[Slide]:
        try:
            components = hitobject.split(',')
            start_pos = Position(int(components[0]), int(components[1]))
            timestamp = int(components[2])
            is_start = is_start
            body_ve = int(components[4])
            slide_type, body_pos, end_pos = HitObjectParser.parse_slide_posline(components[5])
            tick = float(components[7])
            start_ve, end_ve = HitObjectParser.parse_slide_voice_effect(components[8]) if len(components) > 9 else (0, 0)
            if len(components) == 11:
                hitsounds = HitObjectParser.parse_slide_hitsounds(components[9] + ',' + components[10])
                if hitsounds:
                    return Slide(SlidePosition(start_pos, end_pos, body_pos),
                                 timestamp, slide_type, is_start, tick,
                                 SlideVoiceEffect(start_ve, body_ve, end_ve),
                                 hitsounds)
            return Slide(SlidePosition(start_pos, end_pos, body_pos),
                         timestamp, slide_type, is_start, tick,
                         SlideVoiceEffect(start_ve, body_ve, end_ve))
        except IndexError:
            logging.warning('Invalid HitObject: {}'.format(hitobject))
            return None

    @staticmethod
    def parse_spin(hitobject: str) -> Optional[Spin]:
        components = hitobject.split(',')
        try:
            timestamp = SpinTimestamp(int(components[2]), int(components[5]))
            voice_effect = int(components[4])
            if len(components) == 7:
                hitsounds = HitObjectParser.parse_circle_spin_hitsounds(components[6])
                if hitsounds:
                    return Spin(timestamp, voice_effect, hitsounds)
            return Spin(timestamp, voice_effect)
        except IndexError:
            logging.warning('Invalid Hitobject: {}'.format(hitobject))
            return None
