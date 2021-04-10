FRAME_ATTRIBUTES = {
    'weather': [
        'rainy',
        'snowy',
        'clear',
        'overcast',
        'undefined',
        'partly cloudy',
        'foggy'
    ],
    'scene': [
        'tunnel',
        'residential',
        'parking lot',
        'undefined',
        'city street',
        'gas stations',
        'highway'
    ],
    'timeofday': [
        'daytime',
        'night',
        'dawn/dusk',
        'undefined'
    ],
}

LABEL_ATTRIBUTES = {
    'occluded': [
        'true',
        'false'
    ],
    'truncated': [
        'true',
        'false'
    ],
    'trafficLightColor': [
        'red',
        'green',
        'yellow',
        'none'
    ],
    # only attribute when category is 'drivable area'
    'areaType': [
        'direct',
        'alternative'
    ],
    # only attributes when category is 'lane'
    'laneDirection': [
        'parallel',
        'vertical'
    ],
    'laneStyle': [
        'solid',
        'dashed'
    ],
    'laneTypes': [
        'crosswalk',
        'double other',
        'double white',
        'double yellow',
        'road curb',
        'single other',
        'single white',
        'single yellow'
    ]
}

LABEL_CATEGORIES = [
    'bike',
    'bus',
    'car',
    'motor',
    'person',
    'rider',
    'traffic light',
    'traffic sign',
    'train',
    'truck',
    'drivable area',
    'lane'
]
