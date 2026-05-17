import pandas as pd
MERGE_MAP = {
    # AC: включить/выключить/режим кондиционера часто формулируются одинаково
    "ac_set": "ac_control",
    "ac_off": "ac_control",
    "ac_mode_set": "ac_control",

    # Radio: выбор радио и выбор станции часто совпадают в формулировках
    "radio_set": "radio_control",
    "radio_set_station": "radio_control",

    # Memory: установка/вызов памяти часто размазаны в синтетических фразах
    "seat_memory_set": "memory_position_control",
    "seat_memory_recall": "memory_position_control",
    "steering_wheel_memory_recall": "memory_position_control",

    # Calls: все действия внутри звонка
    "call_start": "call_control",
    "call_answer": "call_control",
    "call_hangup": "call_control",

    # Messages
    "message_read": "message_control",
    "message_reply": "message_control",
    "message_send": "message_control",

    # Phone connection
    "phone_connect": "phone_connection_control",
    "phone_disconnect": "phone_connection_control",
    "phone_pair": "phone_connection_control",
    "phone_unpair": "phone_connection_control",

    # Front lights: фары/дальний свет часто пересекаются
    "headlights_set": "front_lights_control",
    "high_beam_set": "front_lights_control",

    # Interior lights
    "ambient_light_control": "cabin_light_control",
    "interior_light_set": "cabin_light_control",
    "reading_light_set": "cabin_light_control",

    # Mute: отключение звука навигации и общее mute часто спутаны
    "volume_mute_set": "mute_control",
    "nav_mute_set": "mute_control",

    # Driving modes
    "drive_mode_set": "drive_mode_control",
    "powertrain_mode_set": "drive_mode_control",
    "suspension_mode_set": "drive_mode_control",

    # Cruise control: set/resume/cancel — один функциональный блок
    "cruise_control_set": "cruise_control",
    "cruise_control_resume": "cruise_control",
    "cruise_control_cancel": "cruise_control",
}



data = pd.read_csv("samples/data.csv")
data["intent"] = data["intent"].apply(lambda x: MERGE_MAP.get(x, x))

data.to_csv("samples/new_data.csv", index=False)