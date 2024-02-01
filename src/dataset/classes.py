from typing import Dict, Any 


classId2className = {
                    # Base classes
                    1: 'tree', 
                    2: 'rangeland', 
                    3: 'bareland', 
                    4: 'agric land type 1', 
                    5: 'road type 1', 
                    6: 'sea, lake, & pond', 
                    7: 'building type 1',                                                   
                    # Novel classes (classnames to be updated, see config file 'oem.yaml')
                    8: '', 
                    9: '', 
                    10: '',
                    11: ''
                    }


def update_novel_classes(classId2className, novel_classes: Dict[int, str]) -> None: 
    """    
    input :
        classId2className: Dict
        novel_classes : Dict of novel classes (Id2Name)    
    """
    for k, v in novel_classes.items():
        classId2className[int(k)] = v


def get_classes_split() -> Dict[str, Any]:
    """
    Returns the split of classes
    returns :
         split_classes : Dict.
    """
    classes_split = {'base': list(range(1, 8)), 'val': list(range(8, 12))}
    return classes_split
