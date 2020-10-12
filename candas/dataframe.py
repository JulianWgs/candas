#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataframe to analyze CAN data.
"""

import os
import copy
import glob
import datetime
import hashlib
import numpy as np
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import cantools
import can
from . metadata import MetaData


class CANDataLog(dict):
    """

    This works like a dict from the outside. All signals can be accessed using
    their name via the dictionary accessor.
    Beyond that other function are implemented.
    """

    def __init__(self, log_data, dbc_db, file_hash_blf, file_hash_mat, source,
                 metadata=None):
        """
        Parameters
        ----------
        log_data : :obj:`dict` of ndarray
            The keys of the dictionary are the signals of the data log.
            Each dictionary entry contains the time and value information.
        dbc_db : cantools.db.Database
            The dbc database which was used to convert the data from a binary
            format.
        file_hash_blf : str
            The file hash of the original .blf file.
        file_hash_mat : str
            The file hash of the converted .mat file.
            This is an unique identifiers.
        source : str
            The source from where this object was created.
            It can be either from a file, database or from a fake data
            generator.

        Returns
        -------
        None.

        """
        super(CANDataLog, self).__init__(log_data)
        self.__dbc_db = dbc_db
        if not metadata:
            metadata = dict()
        metadata["file_hash_blf"] = file_hash_blf
        metadata["file_hash_mat"] = file_hash_mat
        metadata["source"] = source
        self.metadata = MetaData(metadata)
        self.__session_id = None

    def __repr__(self):
        return self.__dbc_db.version

    @property
    def dbc_db(self):
        """
        Return the dbc database of the data log.


        Returns
        -------
        cantools.db.Database
            The dbc database which was used to convert the data from a binary
            format.

        """
        return self.__dbc_db

    @property
    def session_id(self):
        """
        Session ID in the database.
        """
        return self.__session_id

    @session_id.setter
    def session_id(self, _):
        """
        Prohibits the changing of the session ID in the database.ng
        """
        raise ValueError("Session id cannot be changed this way."
                         "Use the get_session_id function.")

    def set_metadata(self, metadata_dict):
        """
        Set the necessary metadata to upload to data log to the database.

        Parameters
        ----------
        metadata_dict : :obj:`dict`
            dictionary which contains all values to be set as attributes to
            the class.

        Returns
        -------
        CANDataLog
            Class of data log.

        """
        for key, value in metadata_dict.items():
            self.metadata[key] = value
        return self

    def plot_line(
            self,
            names=None,
            start=0.0, end=0.0, step=1,
            alpha=0.5,
            ax=None):
        """
        Plot a line plot of the datalog and add axis descriptions.

        Parameters
        ----------
        names : :obj:`list` of :obj:`str`, optional
            names of the signals to plot. If no names are given all available
            signals are plotted. This is useful when plotting a data log
            directly after import. The default is None.
        start : float, optional
            Start time of the plot. All values recorded before that time are
            ignored. The default is 0.0.
        end : float, optional
            End time of the plot. All values recorded after that time are
            ignored. To not set the upper limit, set this value to 0.0.
            The default is 0.0.
        step : int, optional
            The steps between plotted data points. To plot every other point
            use the value 2, for every third use 3. The default is 1. This can
            be useful for very frequent signals.
        alpha : float, optional
            Alpha value/opacity of the plot lines. The default is 0.5.
        ax : matplotlib axis, optional
            The axis on which to plot. If its set to None the axis will be set
            to the current axis. The default is None.

        Returns
        -------
        ax : matplotlib axis
            axis with plot data

        """
        # When no names are given all signal are plotted
        # This is useful when you want to method chain loading and plotting
        if names is None:
            names = list(self.keys())

        # find largest time in log_name with key names
        if end == 0.0:
            for name in names:
                if end < self[name][-1][0]:
                    end = self[name][-1][0]

        grouped_names = group_names(names)

        if ax is None:
            ax = plt.gca()
        axes = [ax]
        for _ in range(len(grouped_names) - 1):
            axes.append(ax.twinx())

        count = 0
        for idx, names_in_group in enumerate(grouped_names.values()):
            # iterate through names of signals and plot them from log_data
            for name in names_in_group:
                x_data, y_data, start_idx, end_idx = get_data_for_plotting(
                    self[name], start, end
                )
                axes[idx].plot(
                    x_data[start_idx:end_idx:step],
                    y_data[start_idx:end_idx:step],
                    "C" + str(count % 9),
                    alpha=alpha,
                    label=name,
                )
                count += 1

            axes[idx].set_xlabel("Time [s]")
            axes[idx].set_ylabel(get_label_from_names(names_in_group,
                                                      self.__dbc_db))

            if len(names_in_group) <= 8:
                axes[idx].legend(loc=idx + 1)
            if idx >= 2:
                axes[idx].spines["right"].set_position(
                    ("axes", 1 + 0.15 * idx))
        ax.grid()
        return ax

    def plot_histogram(self, names=None, bins=20, ax=None):
        """Plot histogram of give signal names.

        Parameters
        ----------
        names : :obj:`list` of :obj:`str`, optional
            names of the signals to plot. If no names are given all available
            signals are plotted. This is useful when plotting a data log
            directly after import. The default is None.
        bins : int
            bins parameter from matplotlib
        ax : matplotlib axis, optional
            The axis on which to plot. If its set to None the axis will be set
            to the current axis. The default is None.

        Returns
        -------
        ax : matplotlib axis
            axis with plot data

        """
        # When no names are given all signal are plotted
        # This is useful when you want to method chain loading and plotting
        if names is None:
            names = list(self.keys())
        histogram_values = []
        # iterating over all signal and appending values to list
        for name in names:
            _, values = self[name].T
            histogram_values.append(values)
        if ax is None:
            ax = plt.gca()
        ax.hist(histogram_values, bins=bins)
        ax.legend(names)
        ax.set_ylabel("No. of occurences")
        ax.set_xlabel(get_label_from_names(names, self.__dbc_db))
        return ax

    def plot_bit_signals(self, names=None, colors=("r", "g"), ax=None):
        """Plot signals which can only be on or off.

        Parameters
        ----------
        names : :obj:`list` of :obj:`str`, optional
            Names of the signals to plot. If no names are given all available
            signals are plotted. This is useful when plotting a data log
            directly after import. The default is None.
        colors : :obj:`list` of :obj:`str`, optional
            List of strings/colors. Value need to correspond to a value of
            a matplotlib color. The default is ["r", "g"].
        ax : matplotlib axis, optional
            The axis on which to plot. If its set to None the axis will be set
            to the current axis. The default is None.

        Returns
        -------
        ax : matplotlib axis
            axis with plot data

        """
        # When no names are given all signal are plotted
        # This is useful when you want to method chain loading and plotting
        if names is None:
            names = list(self.keys())
        if ax is None:
            ax = plt.gca()
        for name in names:
            time, values = self[name].T
            # Get all the indices where the signal is changing
            switch_idx = np.argwhere(np.diff(values) != 0)
            # Add zero so the first sate is not missing
            switch_idx = np.insert(switch_idx, 0, 0)
            for start_idx, end_idx in zip(switch_idx[:-1], switch_idx[1:]):
                # plot bar from last change to current one with specific color
                ax.barh(
                    name,
                    width=time[end_idx] - time[start_idx],
                    left=time[start_idx],
                    color=colors[int(values[end_idx])],
                )
            ax.set_xlabel("Time [s]")
        return ax

    def plot_accumulator(self, time, signal_type,
                         min_value=None, max_value=None, ax=None,
                         signal_name=None, sensors_per_stack=None,
                         number_of_stacks=None):
        """
        Plot the sensor values spaced correctly of every accumulator sensor.

        Parameters
        ----------
        time : float
            Time stamp of sensor values. The specific sensor values will be
            interpolated.
        signal_type : str
            Choose between voltage or temperature.
        min_value : float, optional
            Minimum value for the colorbar. The default is None.
        max_value : float, optional
            Maximum value of the colorbar. The default is None.
        ax : matplotlib axis, optional
            The axis on which to plot. If its set to None the axis will be set
            to the current axis. The default is None.
        signal_name : str
            Prefix of the accumulator temperature or voltage signals.
        sensors_per_stack : int
            Count of temperature or voltage sensors in one stack.
        number_of_stacks : int
            Number of stacks in the accumulator.

        Returns
        -------
        ax : matplotlib axis
            axis with plot data

        """
        assert signal_type in ["temperature", "voltage"]

        try:
            if signal_type == "temperature":
                if signal_name is None:
                    signal_name = self.metadata["accumulator_temperature_name"]
                if sensors_per_stack is None:
                    sensors_per_stack = self.metadata["temperature_sensors_"
                                                      "per_stack"]
            elif signal_type == "voltage:":
                if signal_name is None:
                    signal_name = self.metadata["accumulator_voltage_name"]
                if sensors_per_stack is None:
                    sensors_per_stack = self.metadata["voltage_sensors_"
                                                      "per_stack"]

            if number_of_stacks is None:
                number_of_stacks = self.metadata["number_of_stacks"]
        except AttributeError:
            raise AttributeError("Please give signal_name, "
                                 "sensors_per_stack and number_of_stacks "
                                 "as parameter or "
                                 "set it as class attribute")

        sensor_values = convert_sensor_data_to_grid(
            self, signal_name,
            range(number_of_stacks), range(sensors_per_stack), time)

        if ax is None:
            ax = plt.gca()

        im = ax.imshow(
            sensor_values,
            cmap="hot",
            interpolation="nearest",
            vmin=min_value,
            vmax=max_value,
        )
        # Add colorbar to axis like in seaborn
        # https://github.com/mwaskom/seaborn/blob/master/seaborn/matrix.py
        ax.figure.colorbar(im, ax=ax)
        return ax

    def plot_stack(self, stack, time, ax=None,
                   name=None, sensors_per_stack=None):
        """
        Plot sensor value slice of stack at spedific time.

        Parameters
        ----------
        stack : int
            Number of the stack.
        time : float
            Time stamp of sensor values. The specific sensor values will be
            interpolated.
        ax : matplotlib axis, optional
            The axis on which to plot. If its set to None the axis will be set
            to the current axis. The default is None.
        name : str
            Prefix of the accumulator temperature signals.
        sensors_per_stack : int
            Count of temperature sensors in one stack.

        Raises
        ------
        ValueError
            When not yet supported dbc version is used.

        Returns
        -------
        ax : matplotlib axis
            axis with plot data

        """
        try:
            if name is None:
                name = self.metadata["accumulator_temperature_name"]
            if sensors_per_stack is None:
                sensors_per_stack = range(
                    self.metadata["temperature_sensors_per_stack"])
        except AttributeError:
            raise AttributeError("Please give name and  sensors_per_stack "
                                 "as parameter or set it as metadata.")

        stacks = [stack]
        # return values from this function come in a regular grid already
        # reversed for stack position
        sensor_values = convert_sensor_data_to_grid(self,
                                                    name,
                                                    stacks,
                                                    sensors_per_stack,
                                                    time)
        if ax is None:
            ax = plt.gca()

        # reverse values to keep same viewpoint
        if stack % 2 == 0:
            step = -1
            ax.invert_xaxis()
        else:
            step = 1

        ax.plot(sensor_values[0, ::step], label=str(time) + "s")
        ax.set(
            xlabel="Temperature Sensor",
            ylabel="Temperature [°C]",
            title="Stack Temperature Slice (Rear View)",
        )
        ax.legend()
        return ax

    def plot_xy(self, x_signal_name, y_signal_name, ax=None):
        """
        Plot the x, y values of time series as pairs.

        The resulting plot has no time dimension, but only shows the
        relationship between the two signal.

        Parameters
        ----------
        x_signal_name : str
            Name of x signal.
        y_signal_name : str
            Name of y signal.
        ax : matplotlib axis, optional
            The axis on which to plot. If its set to None the axis will be set
            to the current axis. The default is None.

        Returns
        -------
        ax : matplotlib axis
            axis with plot data

        """
        _, x_values, y_values = get_xy_from_timeseries(self[x_signal_name],
                                                       self[y_signal_name])
        if ax is None:
            ax = plt.gca()
        ax.plot(x_values, y_values)
        x_label = "{} [{}]".format(
            x_signal_name, get_label_from_names([x_signal_name], self.__dbc_db)
        )
        y_label = "{} [{}]".format(
            y_signal_name, get_label_from_names([y_signal_name], self.__dbc_db)
        )
        ax.set(xlabel=x_label, ylabel=y_label)
        return ax

    def to_dataframe(self, names=None, mode="concat", frequency=None):
        """
        Convert signals to pandas dataframe.

        Parameters
        ----------
        names : :obj:`list` of :obj:`str`, optional
            Names of the signals to plot. If no names are given all available
            signals are plotted. This is useful when plotting a data log
            directly after import. The default is None.
        mode : str, optional
            The value "concat" calls the to_dataframe_concat function.
            The value "sampling" calss the to_dataframe_sampling function.
            Look at the corresponding functions for more details.
            The default is "co".
        frequency : float, optional
            If mode = "sampling" and sampling frequency must be given.
            Look at the function to_dataframe_sampling for more information.
            The default is None.

        Raises
        ------
        ValueError
            When wrong mode e.g. not "concat" or "sampling" is supplied.

        Returns
        -------
        pd.DataFrame
            The Dataframe has only one time column and all the other time
            columns of the signals are merged in this time column.

        """
        if mode == "concat":
            return self.to_dataframe_concat(names)
        elif mode == "sampling":
            return self.to_dataframe_sampling(names, frequency)
        else:
            raise ValueError("mode must be set to 'concat' or 'sampling'")

    def to_dataframe_concat(self, names=None):
        """
        Convert signals to pandas dataframe by concatenating signals.

        Signals in different messages have different timestamps. Signals in the
        same message have the same timestamp. This function merges all signals
        based on their timestamp. When two signals are in the same message both
        will be in the same row. If two signals are not in the same message
        (have different timestamps) each will keep its timestamp and be on a
        different row. For every timestamp where there is no signal, the value
        be NaN.

        When using this function on signals from a lot of different signals the
        resulting dataframe will have a lot of NaN values. It is recommended to
        use a sparse dataframe:

        https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html

        Notes
        -----
            Memorywise it is not feasible to convert the whole data log in a
            dataframe with this method!

        Parameters
        ----------
        names : :obj:`list` of :obj:`str`, optional
            Names of the signals to conerte. If no names are given all
            available signals are plotted. This is useful when converting a
            data log directly after import. The default is None.

        Returns
        -------
        df : pd.DataFrame
            The Dataframe has only one time column and all the other time
            columns of the signals are merged in this time column.

        """
        if names is None:
            names = list(self.keys())
        df = pd.DataFrame()
        for name in names:
            try:
                # Create a new DataFrame which will be concatenated to
                # the other DataFrame. Before that this DataFrame will be
                # prepared.
                df_concat = pd.DataFrame(self[name], columns=["time", name])
                # Make sure time values are floats. Sometime there can be
                # Strings if values are also strings
                df_concat["time"] = df_concat["time"].astype(float)
                # Set "time" column as index
                df_concat = df_concat.set_index("time")
                # Concat both DataFrame so values with the same index are
                # together in one row. When values don't share the same
                # time a new row with NaN for other values is created
                df = pd.concat(
                    [df, df_concat], sort=False, axis=1
                )
            except KeyError:
                pass
        df = df.reset_index().rename(columns={"index": "time"})
        df.sort_values("time")
        return df

    def to_dataframe_sampling(self, names=None,
                              frequency=None, timestamps=None):
        """
        Convert signals to pandas dataframe by sampling signals.

        This function samples all input signals at the given frequency and
        returns a dataframe with those timestamps and the corresponding values.

        The resulting dataframe does not contain any NaN values (unless in the
        original signal the value is NaN). However by sampling each signal
        the result is less accurate than to_dataframe_concat.


        Parameters
        ----------
        names : :obj:`list` of :obj:`str`, optional
            Names of the signals to convert. If no names are given all
            available signals are converted. This is useful when conveting a
            data log directly after import. The default is None.
        frequency : float, optional
            Freqency in Hz or 1/s by which the signal should be sampled. The
            first time value of the dataframe will be the largest start
            timestamp of all input signals and the last will be the smallest
            end timestamp. The default is 1.0.
        frequency : float, optional
            Freqency in Hz or 1/s by which the signal should be sampled. The
            first time value of the dataframe will be the largest start
            timestamp of all input signals and the last will be the smallest
            end timestamp. This can be used instead of timestamps.
            The default is None.
        timestamps : :obj:`list` of :obj:`str`, optional
            Timestamps where the sampling should happen. This can be used
            instead of frequency.

        Returns
        -------
        df : pd.DataFrame
            The Dataframe has only one time column and all the other time
            columns of the signals are merged in this time column.

        """
        if names is None:
            names = list(self.keys())
        if timestamps is None:
            # get the time span of data
            # important: smallest time span of all signals is used
            time_min, time_max = 0, np.infty
            for name in names:
                time_min_local = self[name][0][0]
                time_max_local = self[name][-1][0]
                if time_min_local > time_min:
                    time_min = time_min_local
                if time_max_local < time_max:
                    time_max = time_max_local

            # set global variables
            length = int((time_max - time_min) * frequency)
            timestamps = np.linspace(time_min, time_max, length)

        dataframe = pd.DataFrame(timestamps, columns=["time"])
        for name in names:
            time, values = self[name].T
            dataframe[name] = np.interp(timestamps, time, values)
        return dataframe

    def get_can_stats_for_signals(self, signal_names):
        """
        Get different stats of the time between two signal.

        Parameters
        ----------
        signal_names : :obj:`list` of :obj:`str`
            Contains the signal on which the return info is based.
            Contains pairs of (time, value).

        Returns
        -------
        pd.DataFrame
            DataFrame with the stats.

        """
        stats_dict = {"name": [], "mean": [], "std": [], "min": [], "max": []}
        for signal_name in signal_names:
            diff = np.diff(self[signal_name][:, 0])
            stats_dict["signal_name"].append(signal_name)
            stats_dict["mean"].append(np.mean(diff))
            stats_dict["std"].append(np.std(diff))
            stats_dict["min"].append(np.min(diff))
            stats_dict["max"].append(np.max(diff))

        return pd.DataFrame(stats_dict)


def from_file(dbc_db, filename, names=None, always_convert=False,
              verbose=True, **kwargs):
    """
    Create CANDataLog object from a .blf (raw binary) or .mat (converted) file.

    Parameters
    ----------
    dbc_db : cantools.db.Database
        The dbc database which was used to convert the data from a binary
        format.
    filename : str
        absolute or relative path of the log file without the extension.
    names : :obj:`list` of :obj:`str`, optional
        Names of the signals to import. If no names are given all
        available signals are imported. This is useful when memory is an issue.
        The default is None. However he full datalog is always converted.
    always_convert : bool
        Set to True to always convert .blf to .mat, also when .mat already
        exists.
    verbose : bool, optional
        Set to False to have no readout. The default is True.

    Raises
    ------
    KeyError
        When one of the names does not correspond to a signal name in the log.

    Returns
    -------
    CANDataLog
        Class of data log.

    """
    if os.path.isfile(filename + ".mat") and not always_convert:
        if verbose:
            print("Using already converted data")
        # add timer
        log_data = loadmat(filename + ".mat")
    else:
        log_data = can.BLFReader(filename + ".blf")
        if verbose:
            print(
                "Converting data to readable format... "
                "this might take several minutes"
            )
        log_data = decode_data(log_data, dbc_db)
        savemat(filename + ".mat", log_data)

    # Hash file for the record
    hasher = hashlib.sha256()
    if os.path.isfile(filename + ".blf"):
        with open(filename + ".blf", "rb") as file:
            buf = file.read()
            hasher.update(buf)
        file_hash_blf = hasher.hexdigest()
    else:
        # Give fake file hash
        file_hash_blf = ("00000000000000000000000000000000"
                         "00000000000000000000000000000000")
    with open(filename + ".mat", "rb") as file:
        buf = file.read()
        hasher.update(buf)
    file_hash_mat = hasher.hexdigest()

    # return full data log
    if names is None:
        ret_value = log_data
    # return only given names
    else:
        ret_value = {}
        for name in names:
            try:
                ret_value[name] = log_data[name]
            except KeyError:
                raise KeyError("Wrong name: ", name)

    return CANDataLog(ret_value, dbc_db,
                      file_hash_blf, file_hash_mat, source="file", **kwargs)


def from_fake(dbc_db,
              signals_properties,
              file_hash_blf=("00000000000000000000000000000000"
                             "00000000000000000000000000000000"),
              file_hash_mat=("00000000000000000000000000000000"
                             "00000000000000000000000000000000"), **kwargs):
    """
    Create a data log with propterties given in a list of dicts with key name
    arguments of create_fake_can_data function.

    Parameters
    ----------
    dbc_db : cantools.db.Database
        The dbc database which was used to convert the data from a binary
        format.
    signals_properties : :obj:`dict`
        Key-Value pairs of the properties. See create_fake_can_data for more
        information.
    file_hash_blf : str, optional
        Fake hash of the .blf file.
        The default is ("00000000000000000000000000000000"
        "00000000000000000000000000000000").
    file_hash_mat : str, optional
        Fake hash of the .mat file.
        The default is ("00000000000000000000000000000000"
        "00000000000000000000000000000000").

    Returns
    -------
    CANDataLog
        Class of data log.

    """
    log_data = {}
    # To prevent that .pop remove entry from original dict
    signals_properties = copy.deepcopy(signals_properties)
    for signal_properties in signals_properties:
        name = signal_properties.pop("name")
        log_data[name] = create_fake_can_data(**signal_properties)

    return CANDataLog(log_data, dbc_db,
                      file_hash_blf, file_hash_mat, source="fake", **kwargs)


def load_dbc(folder, verbose=True):
    """
    Load all dbc files from specified folder add to dbc database.

    Parameters
    ----------
    folder : str
        Absolute or relative path to folder, which contains dbc files.
    verbose : bool, optional
        Set to False to have no readout. The default is True.

    Returns
    -------
    dbc_db : cantools.db.Database
        dbc database to convert the data from binary format.

    """
    dbc_db = cantools.db.Database(
        messages=None, nodes=None, buses=None, version=None
    )
    if verbose:
        print("Loading dbc...")
    dbc_files = glob.iglob(folder + "/*.dbc")
    file_count = 0
    for dbc_file in dbc_files:
        file_count += 1
        dbc_db.add_dbc_file(dbc_file)
    assert file_count > 0, "No dbc-files in '{}'!".format(folder)
    if verbose:
        print("Finished loading.")
    return dbc_db


def decode_data(log, dbc_db):
    """
    Decode byte-object to readable log file.

    Parameters
    ----------
    log : can.io.blf.BLFReader
        File object of raw blf data.
    dbc_db : cantools.db.Database
        dbc database to convert the data from binary format.

    Returns
    -------
    decoded : :obj:`dict`
        Dictionary of numpy arrays. The first row is the time data and the
        second row is the value data.

    """
    decoded = {}
    undecoded = []
    for msg in log:
        try:
            dec = dbc_db.decode_message(msg.arbitration_id, msg.data)
            if dec:
                for key, data in dec.items():
                    if key not in decoded:
                        decoded[key] = []
                    decoded[key].append([msg.timestamp, data])
        # Catch every error which can occur and save the msg
        except:
            undecoded.append(msg)

    error_ids = set(map(lambda m: m.arbitration_id, undecoded))
    if error_ids:
        print("The following IDs caused errors: " + str(error_ids))

    return decoded


def get_xy_from_timeseries(x_data, y_data):
    """
    Return the x, y values of time series as pairs.

    Parameters
    ----------
    x_data : numpy.ndarray
        Numpy arrays, where the first row is the time data and the
        second row is the value data.
    y_data : numpy.ndarray
        Numpy arrays, where the first row is the time data and the
        second row is the value data.

    Returns
    -------
    :obj:`tuple` of numpy.ndarray
        Tuple which contains the unified timestamp, one original value signal
        and one interpolated value signal.

    """
    t_x, data_x = x_data.T
    t_y, data_y = y_data.T

    # data_y(t_x)
    if len(t_x) < len(t_y):
        data_y = interp1d(t_y, data_y, fill_value="extrapolate")(t_x)
        time = t_x
    # data_x(t_y)
    elif len(t_x) > len(t_y):
        data_x = interp1d(t_x, data_x, fill_value="extrapolate")(t_y)
        time = t_y
    # just two equal long arrays but not equal in content
    # preference of time values in x_data
    elif (t_x != t_y).any():
        data_y = interp1d(t_y, data_y, fill_value="extrapolate")(t_x)
        time = t_x
    # Signals in the same message
    elif (t_x == t_y).all():
        time = t_x  # could also be t_y

    return time, data_x, data_y


def convert_sensor_data_to_grid(log_data, name, stacks, sensors, time):
    """
    Convert the sensor data from the accumulator to plot heat maps.

    The data is put onto a grid. The grid is corrected for the alignment of the
    stacks inside the accumulator. With this grid it is possible to plot
    heat maps which are spaced correct.

    Parameters
    ----------
    log_data : CANDataLog
        Class of data log.
    name : str
        Common name of the sensor signal. For example when naming is
        Sensor_1_1, Sensor_1_2 and so on it would be "Sensor".
    stacks : :obj:`list` of int
        List of stack numbers of which the sensor values should be calculated.
    sensors : :obj:`list` of int
        List of sensor numbers of which the sensor values should be calculated.
    time : float
        Timestamp for when the values should be calculated.

    Returns
    -------
    sensor_values : numpy.ndarray
        Grid with the sensor values. Dimensions are len(stacks) * len(sensors).

    """
    sensor_values = np.zeros([len(stacks), len(sensors)], dtype=float)

    for idx_stack, stack in enumerate(stacks):
        for idx_sensor, sensor in enumerate(sensors):
            key = name + str(stack + 1) + "_" + str(sensor + 1)
            # reverse values to keep orientation
            if stack % 2 == 0:
                idx_sensor = len(sensors) - sensor - 1
            # This can be vectorized, but speed is not important right now
            sensor_value = interp1d(
                log_data[key][:, 0],
                log_data[key][:, 1],
                fill_value="extrapolate")(time)
            sensor_values[idx_stack, idx_sensor] = sensor_value

    return sensor_values


def get_signals_dict_from_db(dbc_db):
    """
    Get dictionary of signals from db where keys are signal names.

    Parameters
    ----------
    dbc_db : cantools.db.Database
        The dbc database which contains the messages and signals.

    Returns
    -------
    signals : :obj:`dict` of cantools.database.can.signal.Signal
        Dictionary of the signals where keys are signal names.

    """
    signals = {}
    for message in dbc_db.messages:
        for signal in message.signals:
            signals[signal.name] = signal
    return signals


def get_label_from_names(names, dbc_db):
    """
    Get Label[unit] from all names of can signals for an axis on a plot.

    Return only a unit if all names share the same unit type. Otherwise it will
    return an empty list.

    Parameters
    ----------
    names : :obj:`list` of :obj:`str`
        names of the signals to get the labels from.
    dbc_db : cantools.db.Database
        The dbc database which was used to convert the data from a binary
        format. It also contains the unit information of a signal.

    Returns
    -------
    :obj:`list` of str
        Labels for the axis.

    """
    unit2name = {
        "V": "Voltage",
        "A": "Current",
        "W": "Power",
        "Wh": "Energy",
        "As": "Electric Charge",
        "°C": "Temperature",
        "1/min": "Revolution Speed",
        "°": "Angle",
        "%": "Travel",
    }

    # get dictionary with signals
    signals = get_signals_dict_from_db(dbc_db)

    units = []
    for name in names:
        units.append(signals[name].unit)

    # check if all units are equal, then one unified label will be created
    if units.count(units[0]) == len(units):
        # Check if not empty string
        if units[0]:
            unit = units[0]
        else:
            unit = "-"
        # Creating labels
        # Add corresponding name to unit, when there is one available
        if unit in unit2name.keys():
            label = "{} [{}]".format(unit2name[unit], unit)
        else:
            label = unit
    else:
        label = ""

    return label


def group_names(names):
    """
    Group names of signals by similarity of name.

    First all numbers are removed from the signal name. Then identical
    numberless signal names are grouped together.

    This function is used to cluster line on a plot.

    Parameters
    ----------
    names : :obj:`list` of :obj:`str`
        names of the signals to group.

    Returns
    -------
    grouped_names : :obj:`dict` of :obj:`list` of :obj:`str`
        Dictionary with the name without numbers as key and all signal names
        with numbers as list items.

    """
    grouped_names = {}
    for name in names:
        # Remove all numbers from signal name
        name_without_numbers = "".join(i for i in name if not i.isdigit())
        if name_without_numbers in grouped_names.keys():
            grouped_names[name_without_numbers].append(name)
        else:
            grouped_names[name_without_numbers] = [name]

    return grouped_names


def get_data_for_plotting(signal, start, end):
    """
    Get the data plus the start and end time from signal.

    This is important, when the user doesn't want to plot the whole signal,
    because of performance reasons.

    Parameters
    ----------
    signal : numpy.ndarray
        Array of time and value data.
    start : float
        Desired start timestamp.
    end : float
        Desired end timestamp.

    Returns
    -------
    x_data : numpy.ndarray
        time data of the signal.
    y_data : numpy.ndarray
        value data of the signal.
    start : float
        Start timestamp.
    end : float
        End timestamp.

    """
    x_data, y_data = signal.T
    start = np.abs(np.array(x_data) - start).argmin()
    end = np.abs(np.array(x_data) - end).argmin()

    return x_data, y_data, start, end


def create_fake_can_data(start, stop,
                         period=0.01,
                         amplitude=1,
                         offset=0,
                         time_noise=0.05,
                         signal_noise=0.02,
                         signal_type="sin",
                         verbose=True):
    """
    Create fake CAN data for testing.

    This function uses sane default values to make usage easier.

    Parameters
    ----------
    start : float
        Start timestamp.
    stop : float
        End timestamp.
    period : float, optional
        Time period between signals. The default is 0.01.
    amplitude : float, optional
        Amplitude of the value signal. The default is 1.
    offset : float, optional
        Offset of the value signal. The default is 0.
    time_noise : float, optional
        Stanard deviation of the jitter of the time period between signals.
        The default is 0.05.
    signal_noise : float, optional
        Standard deviation of jitter of the value signals. The default is 0.02.
    signal_type : str, optional
        Available options are "sin" and "step". The default is "sin".
    verbose : bool, optional
        Set to False to have no readout. The default is True.

    Raises
    ------
    ValueError
        When a not supported signal_type is used.

    Returns
    -------
    data : numpy.ndarray
        Array of time and value data.

    """
    if verbose:
        if time_noise > 0.1:
            print("Time noise is pretty large (Over 10% of period)")
        if signal_noise > 0.05:
            print("Signal noise is pretty large (Over 5% of amplitude)")

    steps = int((stop - start) / period)
    time = np.linspace(start, stop, steps)

    # create different signals
    if signal_type == "sin":
        signal = np.sin(time)
    elif signal_type == "step":
        signal = np.less(np.sin(time), 0.5)
    else:
        raise ValueError("The signal type '{}' is "
                         "not supported.".format(signal_type))

    # adding offset and noise
    signal += offset
    signal += np.random.normal(scale=signal_noise * amplitude,
                               size=len(signal))

    time += np.random.normal(scale=time_noise * period, size=len(time))

    # check if time is steadily growing like in reality
    for idx in range(len(time) - 1):
        if time[idx] >= time[idx + 1]:
            if verbose:
                print("Time not steadily growing -> Fixing")
            time[idx + 1] = time[idx] + time_noise * period * 0.1

    data = np.array(tuple(zip(time, signal)))

    return data
