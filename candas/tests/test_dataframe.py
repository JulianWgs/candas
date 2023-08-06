"""Unit tests for the dataframe module"""
import os
import cantools
import numpy as np
import pandas as pd
import datetime
import candas as cd
import pytest
import sqlite3
import itertools
from candas.dataframe import get_xy_from_timeseries, get_label_from_names


DIR = ""  # for local testing; dont use else! Missing in coverage report
if os.path.exists("/data/"):
    DIR = "/data/"  # for testing on server

dbc_folder = "can/LR17"


def test_can_datalog_properties():
    """Test setting and getting of all properties"""
    dbc_db = cd.load_dbc(dbc_folder)
    signals_properties = [{
        "name": "AMS_Voltage_1_3",
        "start": 2,
        "period": 0.1,
        "stop": 200,
        "signals": [{
            "kind": "float",
            "name": "AMS_CellVoltage11",
        }]
    }]
    file_hash_blf = ("11111111111111111111111111111111"
                     "11111111111111111111111111111111")
    file_hash_mat = ("22222222222222222222222222222222"
                     "22222222222222222222222222222222")
    log_data = cd.from_fake(dbc_db, signals_properties)
    assert repr(log_data)
    assert log_data.dbc_db == dbc_db
    with pytest.raises(ValueError):
        log_data.metadata["source"] = "database"
    with pytest.raises(ValueError):
        log_data.metadata["file_hash_blf"] = file_hash_blf
    with pytest.raises(ValueError):
        log_data.metadata["file_hash_mat"] = file_hash_mat
    with pytest.raises(ValueError):
        log_data.metadata["session_id"] = 1337
    name = "Test"
    log_data.name = name
    assert log_data.name == name
    dbc_commit_hash = "d7bfb79c958932b4490910105883f068008ce236"
    log_data.dbc_commit_hash = dbc_commit_hash
    assert log_data.dbc_commit_hash == dbc_commit_hash
    with pytest.raises(AssertionError):
        log_data.metadata["dbc_commit_hash"] = "abcdef"
    with pytest.raises(AssertionError):
        log_data.metadata["dbc_commit_hash"] = 42
    date = datetime.date(2018, 5, 17)
    log_data.metadata["date"] = date
    assert log_data.metadata["date"] == date
    with pytest.raises(AssertionError):
        log_data.metadata["date"] = "17.05.2018"
    start_time = datetime.time(17, 42)
    log_data.metadata["start_time"] = start_time
    assert log_data.metadata["start_time"] == start_time
    with pytest.raises(AssertionError):
        log_data.metadata["start_time"] = "17:42"
    description = "This is a description"
    log_data.metadata["description"] = description
    log_data.metadata["description"] == description
    with pytest.raises(AssertionError):
        log_data.metadata["description"] = 1337
    with pytest.raises(AssertionError):
        log_data.metadata["description"] = "a" * 141
    event = "FSG"
    log_data.metadata["event"] = event
    assert log_data.metadata["event"] == event
    with pytest.raises(AssertionError):
        log_data.metadata["event"] = 1337
    with pytest.raises(AssertionError):
        log_data.metadata["event"] = "a" * 31
    location = "Braunschweig"
    log_data.metadata["location"] = location
    assert log_data.metadata["location"] == location
    with pytest.raises(AssertionError):
        log_data.metadata["location"] = 1337
    with pytest.raises(AssertionError):
        log_data.metadata["location"] = "a" * 31
    signals_properties = [{
        "name": "AMS_CellTemperature1",
        "start": 3,
        "period": 0.1,
        "stop": 100,
        "signals": [{
            "kind": "float",
            "name": "AMS_Temperature_1_3",
        }]
    }]
    log_data = cd.from_fake(dbc_db, signals_properties)
    log_data.set_metadata({"name": name,
                           "dbc_commit_hash": dbc_commit_hash,
                           "date": date,
                           "description": description,
                           "event": event,
                           "location": location})
    assert log_data.metadata["name"] == name
    assert log_data.metadata["dbc_commit_hash"] == dbc_commit_hash
    assert log_data.metadata["date"] == date
    assert log_data.metadata["description"] == description
    assert log_data.metadata["event"] == event
    assert log_data.metadata["location"] == location

def test_equality():
    # Time and signal noise must be zero for both to be identical
    dbc_db = cd.load_dbc(dbc_folder)
    signals_properties = [{
        "name": "AMS_Voltage_1_3",
        "start": 2,
        "period": 0.1,
        "stop": 200,
        "time_noise": 0,
        "signals": [{
            "kind": "float",
            "name": "AMS_CellVoltage11",
            "signal_noise": 0,
        }]
    }]
    log_data = cd.from_fake(dbc_db, signals_properties)
    assert log_data == log_data

    signals_properties_other = [{
        "name": "AMS_Voltage_1_3",
        "start": 2,
        "period": 0.1,
        "stop": 200,
        "time_noise": 0,
        "signals": [{
            "kind": "float",
            "name": "AMS_CellVoltage11",
            "signal_noise": 0,
        }]
    }]
    log_data_other = cd.from_fake(dbc_db, signals_properties_other)
    assert log_data == log_data_other

    # DataFrame not identically labeled
    signals_properties_different = [{
        "name": "AMS_Voltage_1_3",
        "start": 2,
        "period": 0.2,
        "stop": 200,
        "signals": [{
            "kind": "float",
            "name": "AMS_CellVoltage11",
        }]
    }]
    log_data_different = cd.from_fake(dbc_db, signals_properties_different)
    assert log_data != log_data_different

    # DataFrame identically labeled (no time noise)
    signals_properties_different = [{
        "name": "AMS_Voltage_1_3",
        "start": 2,
        "period": 0.1,
        "stop": 200,
        "time_noise": 0,
        "signals": [{
            "kind": "float",
            "name": "AMS_CellVoltage11",
            "amplitude": 10,
            "offset": 0,
        }]
    }]
    log_data_different = cd.from_fake(dbc_db, signals_properties_different)
    assert log_data != log_data_different


def test_io(tmp_path):
    dbc_db = cd.load_dbc(dbc_folder)
    log_data = cd.from_blf(dbc_db, DIR + "test_small.blf")

    conn_write = sqlite3.connect(tmp_path / "test_small.sql")

    log_data.to_parquet(tmp_path / "test_small")
    log_data.to_mat(tmp_path / "test_small.mat")
    log_data.to_sql(conn_write)

    conn_read = sqlite3.connect(tmp_path / "test_small.sql")

    log_data_parquet = cd.from_parquet(dbc_db, tmp_path / "test_small")
    log_data_mat = cd.from_mat(dbc_db, tmp_path / "test_small.mat")
    log_data_sql = cd.from_sql(dbc_db, conn_read, log_data.messages)

    for left, right in itertools.combinations([log_data, log_data_parquet, log_data_mat, log_data_sql], 2):
        assert left == right


def test_load_dbc():
    # Correct usage
    assert isinstance(cd.load_dbc(dbc_folder), cantools.db.Database)
    assert isinstance(cd.load_dbc(dbc_folder, verbose=False), cantools.db.Database)
    # Give folder parameter with no container .dbc files
    with pytest.raises(AssertionError):
        cd.load_dbc("abc")


def test_create_log_data():
    dbc_db = cd.load_dbc(dbc_folder)
    filename = DIR + "test_small.blf"[:-4]
    assert cd.from_file(dbc_db, filename)[("AMS_CellVoltage11", "AMS_Voltage_1_3")].index[0] == 1.674295
    assert cd.from_file(dbc_db, filename, names=[("AMS_CellVoltage11", "AMS_Voltage_1_3")])[("AMS_CellVoltage11", "AMS_Voltage_1_3")].index[0] == 1.674295
    # TODO: Add test case for when names is a message
    with pytest.raises(KeyError):
        cd.from_file(dbc_db, filename, names=[("dsfjk", "gsdfjh")])
    with pytest.raises(ValueError):
        cd.from_file(dbc_db, filename, names=["gsdfjh"])


    def test_plot_line():
        """Test for different input values"""
        dbc_db = cd.load_dbc(dbc_folder)
        signals_properties = [{
            "name": "AMS_CellVoltage11",
            "start": 2,
            "period": 0.1,
            "stop": 200,
            "signals": [{
                "name": "AMS_Voltage_1_3",
                "kind": "float",
            }]
        }]
        log_data = cd.from_fake(dbc_db, signals_properties)
        assert log_data.plot_line(["AMS_Voltage_1_3"])
        assert log_data.plot_line()
        signals_properties = [
            {
                "name": "AMS_CellVoltage11",
                "start": 2,
                "period": 0.1,
                "stop": 200,
                "signals": [{
                    "name": "AMS_Voltage_1_3",
                "kind": "float",
                }]
            }, {
                "name": "AMS_CellTemperature1",
                "start": 3,
                "period": 0.1,
                "stop": 100,
                "signals": [{
                    "name": "AMS_Temperature_1_3",
                    "kind": "float",
                }]
            }]
        # Test twinx
        log_data = cd.from_fake(dbc_db, signals_properties)
        assert log_data.plot_line()
        signals_properties = [
            {
                "name": "AMS_CellVoltage11",
                "start": 2,
                "period": 0.1,
                "stop": 200,
                "signals": [{
                    "name": "AMS_Voltage_1_3",
                "kind": "float",
                }]
            }, {
                "name": "AMS_CellTemperature1",
                "start": 3,
                "period": 0.1,
                "stop": 100,
                "signals": [{
                    "name": "AMS_Temperature_1_3",
                    "kind": "float",
                }]
            }, { 
                "name": "PCU_ThrottleBrake",
                "start": 2.5,
                "period": 0.1,
                "stop": 105,
                "signals": [{
                    "name": "PCU_BrakeTravel",
                    "kind": "float",
                }]
            }]
        # Test twinx with two axis on one side
        log_data = cd.from_fake(dbc_db, signals_properties)
        assert log_data.plot_line()


def test_plot_histogram():
    dbc_db = cd.load_dbc(dbc_folder)
    signals_properties = [{
        "name": "AMS_CellVoltage11",
        "start": 2,
        "stop": 200,
        "period": 0.1,
        "signals": [{
            "name": "AMS_Voltage_1_3",
             "kind": "float",
        }]
    }]
    log_data = cd.from_fake(dbc_db, signals_properties)
    assert log_data.plot_histogram(["AMS_Voltage_1_3"])
    assert log_data.plot_histogram()


def test_plot_categorical():
    dbc_db = cd.load_dbc(dbc_folder)
    signal_properties = [{
        "start": 0.0,
        "stop": 5.0,
        "period": 0.1,
        "name": "PCU_Status",
        "signals": [{
            "name": "PCU_Status_Steering",
            "kind": "categorical",
        }],
    }]
    log_data = cd.from_fake(dbc_db, signal_properties)
    assert log_data.plot_categorical(["PCU_Status_Steering"])
    assert log_data.plot_categorical()


def test_plot_accumulator():
    dbc_db = cd.load_dbc(dbc_folder)
    signals_properties = list()
    for x in range(9):
        signals_properties.append({
            "name": "AMS_CellTemperature{}".format(x+1),
            "start": 2,
            "stop": 200,
            "period": 0.1,
            "signals": [],
        })
        for y in range(15):
            signals_properties[-1]["signals"].append({
                "name": "AMS_Temperature_{}_{}".format(x+1, y+1),
                "kind": "float",
            })
    log_data = (cd
                .from_fake(dbc_db, signals_properties)
                .set_metadata({
                    "accumulator_temperature_name": "AMS_Temperature_",
                    "temperature_sensors_per_stack": 8,
                    "number_of_stacks": 9,
                }))
    assert log_data.plot_accumulator(10, "temperature")
    with pytest.raises(AssertionError):
        log_data.plot_accumulator(10, "temperatures")


def test_plot_stack():
    dbc_db = cd.load_dbc(dbc_folder)
    stacks = [3, 6]
    signals_properties = list()
    for stack in stacks:
        signals_properties.append({
            "name": "AMS_CellTemperature{}".format(stack+1),
            "start": 2,
            "stop": 200,
            "period": 0.1,
            "signals": [],
        })
        for k in range(8):
            signals_properties[-1]["signals"].append({
                "name": "AMS_Temperature_{}_{}".format(stack+1, k+1),
                "kind": "float",
            })
    log_data = (cd.
                from_fake(dbc_db, signals_properties)
                .set_metadata({
                    "accumulator_temperature_name": "AMS_Temperature_",
                    "temperature_sensors_per_stack": 8,
                }))
    assert log_data.plot_stack(stack=stacks[0], time=2)
    assert log_data.plot_stack(stack=stacks[1], time=2)


def test_plot_xy():
    dbc_db = cd.load_dbc(dbc_folder)
    signals_properties = [{
        "name": "AMS_CellVoltage11",
        "start": 2,
        "stop": 10,
        "period": 0.2,
        "signals": [{
            "name": "AMS_Voltage_1_3",
            "kind": "float",
        }] 
    }, {
        "name": "AMS_CellTemperature1",
        "start": 1,
        "stop": 11,
        "period": 0.3,
        "signals": [{
            "name": "AMS_Temperature_1_5",
            "kind": "float",
        }] 
    }]
    log_data = cd.from_fake(dbc_db, signals_properties)
    assert log_data.plot_xy(
        signals_properties[0]["signals"][0]["name"],
        signals_properties[1]["signals"][0]["name"],
    )


def test_get_xy_from_timeseries():
    x_input = pd.Series(index=np.linspace(5, 10, 100), data=np.linspace(0, 5, 100))
    y_input = pd.Series(index=np.linspace(5, 10, 100), data=np.linspace(0, 2, 100))
    time, x_ret, y_ret = get_xy_from_timeseries(x_input, y_input)
    assert tuple(x_ret) == tuple(np.linspace(0, 5, 100))
    assert tuple(y_ret) == tuple(np.linspace(0, 2, 100))


def test_get_label_from_names():
    names = ["AMK1_ActualVelocity"]
    dbc_db = cd.load_dbc(dbc_folder)
    assert get_label_from_names(names, dbc_db) == "Revolution Speed [1/min]"


def test_to_dataframe():
    dbc_db = cd.load_dbc(dbc_folder)
    signals_properties = [{
        "name": "AMS_CellVoltage11",
        "start": 2,
        "stop": 10,
        "period": 0.2,
        "signals": [{
            "name": "AMS_Voltage_1_3",
            "kind": "float",
        }] 
    }, {
        "name": "AMS_CellTemperature1",
        "start": 1,
        "stop": 11,
        "period": 0.3,
        "signals": [{
            "name": "AMS_Temperature_1_5",
            "kind": "float",
        }] 
    }]
    log_data = cd.from_fake(dbc_db, signals_properties)
    assert isinstance(log_data.to_dataframe([
        signals_properties[0]["signals"][0]["name"],
        signals_properties[1]["signals"][0]["name"],
    ]), pd.DataFrame)
    isinstance(log_data.to_dataframe([
        signals_properties[0]["signals"][0]["name"],
        signals_properties[1]["signals"][0]["name"]
    ], mode="sampling", frequency=2), pd.DataFrame)
    with pytest.raises(ValueError):
        log_data.to_dataframe(
            [signals_properties[0]["name"], signals_properties[1]["name"]],
            mode="abc",
        )
