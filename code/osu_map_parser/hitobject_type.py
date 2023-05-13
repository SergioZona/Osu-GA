from typing import NamedTuple, List


class Position(NamedTuple):
    x: int
    y: int


class Hitsounds(NamedTuple):
    sampleset: int
    additions: int


class SlidePosition(NamedTuple):
    start: Position
    end: Position
    body: List[Position] = []

    def __repr__(self):
        return f'\n\t\tstart={self.start}\n\t\tbody={self.body}\n\t\tend={self.end}\n\t'


class SlideVoiceEffect(NamedTuple):
    start: int = 0
    body: int = 0
    end: int = 0

    def __repr__(self):
        return f'\n\t\tstart={self.start}\n\t\tbody={self.body}\n\t\tend={self.end}\n\t'


class SlideHitsounds(NamedTuple):
    start: Hitsounds = Hitsounds(0, 0)
    body: Hitsounds = Hitsounds(0, 0)
    end: Hitsounds = Hitsounds(0, 0)

    def __repr__(self):
        return f'\n\t\tstart={self.start}\n\t\tbody={self.body}\n\t\tend={self.end}\n\t'


class SpinTimestamp(NamedTuple):
    start: int
    end: int


class Circle(NamedTuple):
    pos: Position
    timestamp: int
    is_start: bool
    voice_effect: int = 0
    hitsounds: Hitsounds = Hitsounds(0, 0)

    def __repr__(self):
        return f'<Circle\tpos={self.pos}\n\ttimestamp={self.timestamp}\n\tis_start={self.is_start}\n\tvoice_effect={self.voice_effect}\n\thitsounds={self.hitsounds}>'


class Slide(NamedTuple):
    pos: SlidePosition
    timestamp: int
    slide_type: str
    is_start: bool
    tick: float
    voice_effect: SlideVoiceEffect = SlideVoiceEffect(0, 0, 0)
    hitsounds: SlideHitsounds = SlideHitsounds(Hitsounds(0, 0), Hitsounds(0, 0), Hitsounds(0, 0))

    def __repr__(self):
        return f'<Slide\tpos=SlidePosition({self.pos})\n\ttimestamp={self.timestamp}\n\tslide_type={self.slide_type}\n\tis_start={self.is_start}\n\ttick={self.tick}\n\tvoice_effect=SlideVoiceEffect({self.voice_effect})\n\thitsounds=SlideHitsounds({self.hitsounds})>'


class Spin(NamedTuple):
    timestamp: SpinTimestamp
    voice_effect: int = 0
    hitsounds: Hitsounds = Hitsounds(0, 0)

    def __repr__(self):
        return f'<Spin\ttimestamp={self.timestamp}\n\tvoice_effect={self.voice_effect}\n\thitsounds={self.hitsounds}>'
