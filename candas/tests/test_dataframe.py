"""Unit tests for the dataframe module"""
import os
import unittest
import cantools
import numpy as np
import pandas as pd
import datetime
import candas as cd
from candas.dataframe import get_xy_from_timeseries, get_label_from_names


DIR = ""  # for local testing; dont use else! Missing in coverage report
if os.path.exists("/data/"):
    DIR = "/data/"  # for testing on server

dbc_folder = "can/LR17"


class TestCANDataLog(unittest.TestCase):
    """Test CANDataLog."""
    def test_properties(self):
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
        self.assertTrue(repr(log_data))
        self.assertEqual(log_data.dbc_db,
                         dbc_db)
        self.assertRaises(ValueError,
                          log_data.metadata.__setitem__, "source", "database")
        self.assertRaises(ValueError,
                          log_data.metadata.__setitem__,  "file_hash_blf", file_hash_blf)
        self.assertRaises(ValueError,
                          log_data.metadata.__setitem__, "file_hash_mat", file_hash_mat)
        self.assertRaises(ValueError,
                          log_data.metadata.__setitem__, "session_id", 1337)
        name = "Test"
        log_data.name = name
        self.assertEqual(log_data.name, name)
        dbc_commit_hash = "d7bfb79c958932b4490910105883f068008ce236"
        log_data.dbc_commit_hash = dbc_commit_hash
        self.assertEqual(log_data.dbc_commit_hash, dbc_commit_hash)
        self.assertRaises(AssertionError,
                          log_data.metadata.__setitem__, "dbc_commit_hash", "abcdef")
        self.assertRaises(AssertionError,
                          log_data.metadata.__setitem__, "dbc_commit_hash", 42)
        date = datetime.date(2018, 5, 17)
        log_data.metadata["date"] = date
        self.assertEqual(log_data.metadata["date"], date)
        self.assertRaises(AssertionError,
                          log_data.metadata.__setitem__, "date", "17.05.2018")
        start_time = datetime.time(17, 42)
        log_data.metadata["start_time"] = start_time
        self.assertEqual(log_data.metadata["start_time"], start_time)
        self.assertRaises(AssertionError,
                          log_data.metadata.__setitem__, "start_time", "17:42")
        description = "This is a description"
        log_data.metadata["description"] = description
        self.assertEqual(log_data.metadata["description"], description)
        self.assertRaises(AssertionError,
                          log_data.metadata.__setitem__, "description", 1337)
        self.assertRaises(AssertionError,
                          log_data.metadata.__setitem__, "description", "a" * 141)
        event = "FSG"
        log_data.metadata["event"] = event
        self.assertEqual(log_data.metadata["event"], event)
        self.assertRaises(AssertionError,
                          log_data.metadata.__setitem__, "event", 1337)
        self.assertRaises(AssertionError,
                          log_data.metadata.__setitem__, "event", "a" * 31)
        location = "Braunschweig"
        log_data.metadata["location"] = location
        self.assertEqual(log_data.metadata["location"], location)
        self.assertRaises(AssertionError,
                          log_data.metadata.__setitem__, "location", 1337)
        self.assertRaises(AssertionError,
                          log_data.metadata.__setitem__, "location", "a" * 31)
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
        self.assertEqual(log_data.metadata["name"], name)
        self.assertEqual(log_data.metadata["dbc_commit_hash"], dbc_commit_hash)
        self.assertEqual(log_data.metadata["date"], date)
        self.assertEqual(log_data.metadata["description"], description)
        self.assertEqual(log_data.metadata["event"], event)
        self.assertEqual(log_data.metadata["location"], location)

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

    log_data.to_parquet(tmp_path / "test_small")
    log_data.to_mat(tmp_path / "test_small.mat")

    log_data_parquet = cd.from_parquet(dbc_db, tmp_path / "test_small")
    log_data_mat = cd.from_mat(dbc_db, tmp_path / "test_small.mat")

    assert log_data == log_data_parquet
    assert log_data == log_data_mat
    assert log_data_parquet == log_data_mat


class TestLoadDBC(unittest.TestCase):
    """Test load_dbc."""
    def test_values(self):
        """Test for different input values"""
        # Correct usage
        self.assertIsInstance(cd.load_dbc(dbc_folder),
                              cantools.db.Database)
        self.assertIsInstance(cd.load_dbc(dbc_folder, verbose=False),
                              cantools.db.Database)
        # Give folder parameter with no container .dbc files
        self.assertRaises(AssertionError, cd.load_dbc, "abc")


class TestCreateLogData(unittest.TestCase):
    """Test create_log_data."""
    def test_values(self):
        """Test for different input values"""
        dbc_db = cd.load_dbc(dbc_folder)
        filename = DIR + "test_small.blf"[:-4]
        self.assertEqual(
            cd.from_file(dbc_db, filename)[("AMS_CellVoltage11", "AMS_Voltage_1_3")].index[0],
            1.674295)
        self.assertEqual(
            cd.from_file(dbc_db, filename, names=[("AMS_CellVoltage11", "AMS_Voltage_1_3")])
            [("AMS_CellVoltage11", "AMS_Voltage_1_3")].index[0],
            1.674295)
        # TODO: Add test case for when names is a message
        self.assertRaises(KeyError, cd.from_file, dbc_db,
                          filename, names=[("dsfjk", "gsdfjh")])
        self.assertRaises(ValueError, cd.from_file, dbc_db,
                          filename, names=["gsdfjh"])


class TestPlotData(unittest.TestCase):
    """Test plot_data."""
    def test_values(self):
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
        self.assertTrue(log_data.plot_line(["AMS_Voltage_1_3"]))
        self.assertTrue(log_data.plot_line())
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
        self.assertTrue(log_data.plot_line())
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
        self.assertTrue(log_data.plot_line())


class TestPlotHistogram(unittest.TestCase):
    """Test plot_histogram."""
    def test_values(self):
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
        self.assertTrue(log_data.plot_histogram(["AMS_Voltage_1_3"]))
        self.assertTrue(log_data.plot_histogram())


class TestPlotCategorical(unittest.TestCase):
    """Test plot_bit_signals."""
    def test_values(self):
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
        self.assertTrue(log_data.plot_categorical(["PCU_Status_Steering"]))
        self.assertTrue(log_data.plot_categorical())


class TestPlotAccumulator(unittest.TestCase):
    """Test plot_accumulator."""
    def test_values(self):
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
        self.assertTrue(log_data.plot_accumulator(10, "temperature"))
        self.assertRaises(AssertionError,
                          log_data.plot_accumulator, 10, "temperatures")


class TestPlotStackTemps(unittest.TestCase):
    """Test plot_stack_temps"""
    def test_values(self):
        """Test for different input values"""
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
        self.assertTrue(log_data.plot_stack(stack=stacks[0], time=2))
        self.assertTrue(log_data.plot_stack(stack=stacks[1], time=2))


class TestPlotXY(unittest.TestCase):
    """Test plot_xy"""
    def test_values(self):
        """Test for different input values"""
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
        self.assertTrue(log_data.plot_xy(signals_properties[0]["signals"][0]["name"],
                                         signals_properties[1]["signals"][0]["name"]))


class TestGetXYFromTimeseries(unittest.TestCase):
    """Test get_xy_from_timeseries"""
    def test_values(self):
        """Test for different input values"""
        x_input = pd.Series(index=np.linspace(5, 10, 100), data=np.linspace(0, 5, 100))
        y_input = pd.Series(index=np.linspace(5, 10, 100), data=np.linspace(0, 2, 100))
        time, x_ret, y_ret = get_xy_from_timeseries(x_input, y_input)
        self.assertSequenceEqual(tuple(x_ret), tuple(np.linspace(0, 5, 100)))
        self.assertSequenceEqual(tuple(y_ret), tuple(np.linspace(0, 2, 100)))


class TestGetLabelFromNames(unittest.TestCase):
    """Test get_label_from_names"""
    def test_values(self):
        """Test for input values"""
        names = ["AMK1_ActualVelocity"]
        dbc_db = cd.load_dbc(dbc_folder)
        self.assertSequenceEqual(get_label_from_names(names, dbc_db),
                                 "Revolution Speed [1/min]")


class TestToDataframe(unittest.TestCase):
    """Test to_dataframe"""
    def test_values(self):
        """Test for different input values"""
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
        # assertIsInstace does not seam to work
        self.assertEqual(type(pd.DataFrame()),
                         type(log_data.to_dataframe(
                             [signals_properties[0]["signals"][0]["name"],
                              signals_properties[1]["signals"][0]["name"]])))
        self.assertEqual(type(pd.DataFrame()),
                         type(log_data.to_dataframe(
                             [signals_properties[0]["signals"][0]["name"],
                              signals_properties[1]["signals"][0]["name"]],
                             mode="sampling",
                             frequency=2)))
        self.assertRaises(ValueError, log_data.to_dataframe,
                          [signals_properties[0]["name"],
                           signals_properties[1]["name"]],
                          mode="abc")
