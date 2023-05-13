import re

from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Tuple, Union


class Colour(NamedTuple):
    R: str
    G: str
    B: str


class MapParser:

    def __init__(self, target_path: str) -> None:
        self.parse_ret = {
            'version': -1,
            'general': {},
            'editor': {},
            'metadata': {},
            'difficulty': {},
            'events': {},
            'timingpoints': {},
            'colours': {},
            'hitobjects': [],
        }  # type: Dict

        # data preparation
        self.__load_file(target_path)
        self.__cut_section()
        self.__get_section()
        self.__transform_sections()
        self.__get_version()

    def __load_file(self, path: str) -> None:
        with open(path, 'r') as fptr:
            lines = fptr.readlines()
        self.lines = [line.strip() for line in lines]

    def __cut_section(self) -> None:
        section_idx_list = []
        key_list = []

        # get cutting point O(n)
        is_first_match = True
        for idx, line in enumerate(self.lines):
            match = re.compile(r'^\[(.*)\]$').match(line)
            if is_first_match and match:
                section_idx_list.append(idx)
                key_list.append(match.group(1).lower())
                is_first_match = False
            elif match:
                section_idx_list.append(idx)
                section_idx_list.append(idx)
                key_list.append(match.group(1).lower())
            elif idx == len(self.lines)-1:
                section_idx_list.append(idx+1)

        if len(key_list) != len(section_idx_list)/2:
            Exception('Failed to load')

        # [1,2,3,4] -> zip([1,3],[2,4]) -> [(1,2), (3,4)]
        idx_pair_list = list(zip(section_idx_list[0:][::2], section_idx_list[1:][::2]))
        # zip(['key1','key2'], [(1,2), (3,4)]) -> [('key1', (1,2)), ('key2', (3,4))]
        key_pair_list = list(zip(key_list, idx_pair_list))

        self.key_list = key_list
        self.sections = OrderedDict(key_pair_list)
        """
            OrderedDict::
            {
                str::'section_a': tuple::(int::start_line, int::end_line),
                ...
            }
        """

    @staticmethod
    def transform_pair_line(line: str) -> Tuple[str, str]:
        line_split = line.split(':')
        key = line_split[0].strip()
        value = line_split[1].strip()
        return (key, value)

    @staticmethod
    def transform_colour_line(line: str) -> Tuple[str, Any]:
        # mypy's bug, the return value should be Tuple[str, Colour]
        line_split = line.split(':')
        key = line_split[0].strip()
        colors = line_split[1].strip().split(',')
        value = Colour(colors[0], colors[1], colors[2])
        return (key, value)

    @staticmethod
    def transform_event_line(line: str) -> Tuple[str, List[str]]:
        # TODO: handle file name such as "aaa,bbb.txt"
        line_split = line.split(',')
        return (line_split[0], line_split[1:])

    @staticmethod
    def transform_timingpoints_line(line: str) -> Tuple[str, List[str]]:
        line_split = line.split(',')
        return (line_split[0], line_split[1:])

    """
    @staticmethod
    def transform_hitobjects_line(line: str) -> Tuple[str, str]:
        line_split = line.split(',')
        return (line_split[2], line_split[0:2]+line_split[3:])
    """

    def __transform_section(self, key: str) -> Union[dict, OrderedDict, List[str]]:
        if key in ['general', 'editor', 'metadata', 'difficulty']:
            tmp_dict = {}
            for item in [MapParser.transform_pair_line(line) for line in self.parse_ret[key] if line != '']:
                tmp_dict[item[0]] = item[1]
            return tmp_dict
        elif key == 'colours':
            tmp_dict = {}
            for item in [MapParser.transform_colour_line(line) for line in self.parse_ret[key] if line != '']:
                tmp_dict[item[0]] = item[1]
            return tmp_dict
        elif key == 'events':
            # TODO: define a OrderedDict with meaningful key, value
            tmp_list = []
            for line in self.parse_ret[key]:
                if '//' in line:
                    if line.split('//')[0]:
                        tmp_list.append(MapParser.transform_event_line(line))
                elif line:
                    tmp_list.append(MapParser.transform_event_line(line))
            return OrderedDict(tmp_list)
        elif key == 'timingpoints':
            # TODO: define a OrderedDict with meaningful key, value
            return OrderedDict([MapParser.transform_timingpoints_line(line) for line in self.parse_ret[key] if line != ''])
        elif key == 'hitobjects':
            return [line for line in self.parse_ret[key] if line != '']
        else:
            raise KeyError('{} is not a valid key'.format(key))

    def __get_section(self, key: Union[str, None]=None) -> None:
        sections = self.sections
        # ignore first line eg: [General] of section => [1:]
        if key and key in sections.keys():
            self.parse_ret[key] = [self.lines[line] for line in range(sections[key][0], sections[key][1])][1:]
        elif not key:
            for key in self.key_list:
                if key in sections.keys():
                    self.parse_ret[key] = [self.lines[line] for line in range(sections[key][0], sections[key][1])][1:]

    def __transform_sections(self, key: Union[str, None]=None) -> None:
        if key:
            self.parse_ret[key] = self.__transform_section(key)
        elif not key:
            for key in self.key_list:
                self.parse_ret[key] = self.__transform_section(key)

    def __get_version(self) -> None:
        if len(self.lines) > 0:
            match = re.compile(r'osu file format v([0-9]*)').match(self.lines[0])
            if match:
                self.parse_ret['version'] = match.group(1)

    @property
    def version(self) -> int:
        return self.parse_ret['version']

    @property
    def general(self) -> Dict[str, str]:
        return self.parse_ret['general']

    @property
    def editor(self) -> Dict[str, str]:
        return self.parse_ret['editor']

    @property
    def metadata(self) -> Dict[str, str]:
        return self.parse_ret['metadata']

    @property
    def difficulty(self) -> Dict[str, str]:
        return self.parse_ret['difficulty']

    @property
    def events(self) -> OrderedDict:
        return self.parse_ret['events']

    @property
    def timingpoints(self) -> OrderedDict:
        return self.parse_ret['timingpoints']

    @property
    def colours(self) -> Dict[str, Colour]:
        return self.parse_ret['colours']

    @property
    def hitobjects(self) -> List[str]:
        return self.parse_ret['hitobjects']
