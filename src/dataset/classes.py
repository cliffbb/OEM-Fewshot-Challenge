from collections import defaultdict
from typing import Dict, Any, List


classId2className = {
                    # Base classes
                    1: 'tree', 
                    2: 'rangeland', 
                    3: 'bareland', 
                    4: 'agric land type 1', 
                    5: 'road type 1', 
                    6: 'sea, lake, & pond', 
                    7: 'building type 1',                                                   
                    # Novel classes 
                    8: 'road type 2', 
                    9: 'river', 
                    10: 'boat & ship',
                    11: 'agric land type 2'
                    }


className2classId = defaultdict(dict)
for id in classId2className:
    className2classId[classId2className[id]] = id


def get_base_classnames() -> List[str]:
    """
    Returns the names of base classes
    returns :
         base_classes : List.
    """
    base_classes = list(classId2className.values())[:7]
    return base_classes


def get_novel_classnames() -> List[str]:
    """
    Returns the names of novel classes for development/evaluation stage
    returns :
         novel_classes : List.
    """
    novel_classes = list(classId2className.values())[7:]
    return novel_classes


def get_classes_split() -> Dict[str, Any]:
    """
    Returns the split of classes
    returns :
         split_classes : Dict.
    """
    classes_split = {'base': list(range(1, 8)), 'val': list(range(8, 12))}
    return classes_split
