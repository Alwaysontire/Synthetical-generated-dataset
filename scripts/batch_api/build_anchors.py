import random


INTENTS = [
    # Climate Control and Windows
    "window_set", "temperature_set", "temperature_change","ac_set", "ac_off", "ac_mode_set", "ac_fan_set", "recirculation_set", "seat_heater_set", "seat_vent_set", "defrost_set",
    "sunroof_open", "sunroof_close",
    # Media Playback
    "play_music", "media_stop", "next_track", "previous_track", "volume_change", "volume_set", "volume_mute_set", "media_source_set", "media_repeat_set",
    "media_shuffle_set", "radio_set", "radio_set_station",
    # Navigation
    "nav_start",
    "nav_cancel",
    "nav_mute_set",          
    # Communication (Phone & Messaging)
    "call_start", "call_answer", "call_hangup", "message_read", "message_reply", "message_send",
    # Vehicle Exterior Controls
    "headlights_set", "high_beam_set", "wipers_set", "wheel_heater_set",
    # Locks and Access
    "door_lock_set", "door_unlock_all", "trunk_open", "trunk_close", "fuel_cap_open", "mirror_fold_set", "mirror_heat_set", "window_child_lock_set", "glovebox_open",
    # Seating & Memory
    "seat_memory_set",
    "seat_memory_recall",
    "steering_wheel_memory_recall",
    # Interior Lighting
    "ambient_light_control",
    "interior_light_set",
    "reading_light_set",
    # Hazard & Indicator Lights
    "hazard_lights_set",
    "fog_lights_set",
    # Dashboard & HUD
    "dashboard_brightness_set",
    "hud_set",
    # Driver Aids & Modes
    "cruise_control_set", "cruise_control_resume", "cruise_control_cancel", "speed_limiter_set", "adaptive_cruise_distance_set", "lane_keep_assist_set", "park_distance_alert_set", "camera_view_set",
    "auto_hold_set", "drive_mode_set", "suspension_mode_set", "regen_level_set", "start_stop_set", "powertrain_mode_set",
    # Vehicle Status Queries
    "battery_status_query", "fuel_range_query", "tire_pressure_query", "odometer_query", "trip_reset", "trip_query", "maintenance_status_query", "warning_explain",
    # System and Miscellaneous
    "phone_pair", "phone_unpair", "phone_connect", "phone_disconnect", "voice_assistant_set", "language_set", "system_volume_set", "system_help", "chitchat"
]



TIMES_RU = ["утро", "день", "вечер", "ночь", "рассвет", "поздний вечер"]
WEATHER_RU = ["дождь", "снег", "туман", "жара", "ветер", "слякоть", "сухо и пыльно"]
ROAD_RU = ["трасса", "город", "пригород", "парковка", "тоннель", "серпантин", "двор"]
SITUATIONS_RU = [
    "шум от шин", "пассажир спит", "дети шумят", "запотело лобовое",
    "запахи с улицы", "радио шипит", "руки мёрзнут", "жарко и душно",
    "входящий звонок", "нужно быстро свернуть", "навигатор путается",
    "ослепляет встречка", "стекло в каплях после мойки", "солнце в лоб"
]
MOODS_RU = ["спокойно", "раздражает шум", "тороплюсь", "хочу тишины", "не хочу отвлекаться"]
BANNED_START_SETS_RU = [
    ["включи", "сделай", "поставь"],
    ["открой", "закрой", "убери"],
    ["давай", "можно", "пожалуйста"],
]
BANNED_VERBS_RU = ["включи", "выключи", "сделай"]

TIMES_EN = ["morning", "afternoon", "evening", "night", "dawn", "late evening"]
WEATHER_EN = ["rain", "snow", "fog", "heat", "wind", "slush", "dry and dusty"]
ROAD_EN = ["highway", "city", "suburbs", "parking lot", "tunnel", "mountain road", "yard"]
SITUATIONS_EN = [
    "tire noise", "passenger sleeping", "kids making noise", "windshield fogged up",
    "smell from outside", "radio static", "hands are cold", "hot and stuffy",
    "incoming call", "need to turn quickly", "GPS confused",
    "oncoming headlights blinding", "wet windshield after car wash", "sun in eyes"
]
MOODS_EN = ["calm", "annoyed by noise", "in a hurry", "want silence", "don't want distractions"]
BANNED_START_SETS_EN = [
    ["turn on", "set", "put on"],
    ["open", "close", "remove"],
    ["let's", "can you", "please"],
]
BANNED_VERBS_EN = ["turn on", "turn off", "set"]


def build_anchor(rand: random.Random, lang: str = "ru") -> str:
    if lang == "en":
        times, weather, road = TIMES_EN, WEATHER_EN, ROAD_EN
        situations, moods = SITUATIONS_EN, MOODS_EN
        banned_sets, banned_verbs = BANNED_START_SETS_EN, BANNED_VERBS_EN
    else:
        times, weather, road = TIMES_RU, WEATHER_RU, ROAD_RU
        situations, moods = SITUATIONS_RU, MOODS_RU
        banned_sets, banned_verbs = BANNED_START_SETS_RU, BANNED_VERBS_RU

    parts = [
        rand.choice(times),
        rand.choice(weather),
        rand.choice(road),
        rand.choice(situations),
        rand.choice(moods),
        rand.choice(rand.choice(banned_sets)),
        rand.choice(banned_verbs)
    ]
    if rand.random() < 0.35:
        parts.append(rand.choice([s for s in situations if s != parts[3]]))
    return ", ".join(parts)