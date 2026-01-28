import random


INTENTS = [
    "window_set","ac_set","ac_off","ac_mode_set","ac_fan_set","seat_heater_set","seat_vent_set",
    "defrost_set","sunroof_set","sunroof_close",
    "play_music","stop_music","media_pause","next_track","previous_track",
    "volume_change","volume_set","volume_mute_set","media_source_set","media_repeat_set","media_shuffle_set",
    "radio_set","radio_set_station",
    "nav_start","nav_cancel",
    "call_start","call_answer","call_hangup",
    "headlights_set","high_beam_set","wipers_set","rain_set","wheel_heater_set",
    "chitchat"
]


TIMES = ["утро", "день", "вечер", "ночь", "рассвет", "поздний вечер"]
WEATHER = ["дождь", "снег", "туман", "жара", "ветер", "слякоть", "сухо и пыльно"]
ROAD = ["трасса", "город", "пригород", "парковка", "тоннель", "серпантин", "двор"]
SITUATIONS = [
    "шум от шин", "пассажир спит", "дети шумят", "запотело лобовое",
    "запахи с улицы", "радио шипит", "руки мёрзнут", "жарко и душно",
    "входящий звонок", "нужно быстро свернуть", "навигатор путается",
    "ослепляет встречка", "стекло в каплях после мойки", "солнце в лоб"
]
MOODS = ["спокойно", "раздражает шум", "тороплюсь", "хочу тишины", "не хочу отвлекаться"]

BANNED_START_SETS = [
    ["включи", "сделай", "поставь"],
    ["открой", "закрой", "убери"],
    ["давай", "можно", "пожалуйста"],
]

BANNED_VERBS = ["включи", "выключи", "сделай"]

def build_anchor(rand: random.Random) -> str:
    parts = [
        rand.choice(TIMES),
        rand.choice(WEATHER),
        rand.choice(ROAD),
        rand.choice(SITUATIONS),
        rand.choice(MOODS),
    ]
    if rand.random() < 0.35:
        parts.append(rand.choice([s for s in SITUATIONS if s != parts[3]]))
    return ", ".join(parts)