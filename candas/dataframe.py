#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataframe to analyze CAN data.
"""

import os
from collections import defaultdict
from collections.abc import Iterable
import copy
import glob
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import datetime
from pathlib import Path
import hashlib
from pyexpat.errors import messages
import numpy as np
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import cantools
import can
from pyarrow import ArrowInvalid
from . metadata import MetaData


class CANDataLog():
    """

    This works like a dict from the outside. All signals can be accessed using
    their name via the dictionary accessor.
    Beyond that other function are implemented.
    """

    def __init__(self, log_data, dbc_db, unique_signal_names=True, metadata=None):
        """
        Parameters
        ----------
        log_data : :obj:`dict` of ndarray
            The keys of the dictionary are the signals of the data log.
            Each dictionary entry contains the time and value information.
        dbc_db : cantools.db.Database
            The dbc database which was used to convert the data from a binary
            format.

        Returns
        -------
        None.

        """
        self._log_data = log_data
        self.messages = list(log_data.keys())
        self.__dbc_db = dbc_db
        if not metadata:
            metadata = dict()
        self.metadata = MetaData(metadata)
        self.__session_id = None

        if unique_signal_names:
            signal_message_mapping = dict()
            for message in dbc_db.messages:
                for signal in message.signals:
                    # Check that signal names are unique
                    # assert signal.name not in signal_message_mapping.keys(), signal.name
                    signal_message_mapping[signal.name] = message.name
        else:
            signal_message_mapping = None
        self.signal_message_mapping = signal_message_mapping

        # TODO: Only works with unique signal names -> Change!
        # INFO:
        # >   self.signals = sum([list(df.columns) for df in self._log_data.values()], start=[])
        # E   AttributeError: 'bytes' object has no attribute 'columns'
        self.signals = sum([list(df.columns) for df in self._log_data.values() if isinstance(df, pd.DataFrame)], start=[])

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            message, signal = key
            if isinstance(message, slice): 
                assert self.signal_message_mapping is not None, "Signal names must be unique if ':' is passed as message name"
                message = self.signal_message_mapping[signal]
            return self._log_data[message][signal]
        else:
            return self._log_data[key]

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

    def __eq__(self, other):
        if isinstance(other, CANDataLog):
            try:
                return (
                    (set(self.messages) == set(other.messages)) and
                    all((self[message] == other[message]).all().all() for message in self.messages)
                )
            except ValueError:
                # Can only compare identically-labeled DataFrame objects
                return False
        return False

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
            names = list(self.signals)

        # find largest time in log_name with key names
        if end == 0.0:
            for name in names:
                if end < self[:, name].index[-1]:
                    end = self[:, name].index[-1]

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
                    self[:, name], start, end
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
            names = list(self.signals)
        histogram_values = []
        # iterating over all signal and appending values to list
        for name in names:
            values = self[:, name].values
            histogram_values.append(values)
        if ax is None:
            ax = plt.gca()
        ax.hist(histogram_values, bins=bins)
        ax.legend(names)
        ax.set_ylabel("No. of occurences")
        ax.set_xlabel(get_label_from_names(names, self.__dbc_db))
        return ax

    def plot_categorical(self, names=None, ax=None, color_mapping=None, legend=True):
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
            names = list(self.signals)
        if ax is None:
            ax = plt.gca()
        if color_mapping is None:
            unique_values = set()
            for name in names:
                unique_values.update(self[:, name].unique())
            color_mapping = {value: f"C{k}" for k, value in enumerate(unique_values)}
        assert len(color_mapping) <= 10, f"Only 10 different categories are possible, but provided {len(color_mapping)}"
        for name in names:
            time, values = self[:, name].index, self[:, name].values
            # TODO: Only necessary, because the time for string signals is also string
            time = time.astype(float)
            # Get all the indices where the signal is changing
            switch_idx = list(np.argwhere(values[:-1] != values[1:]).flatten())
            # Add zero so the first sate is not missing
            switch_idx.insert(0, 0)
            switch_idx.append(len(values) -1)
            for start_idx, end_idx in zip(switch_idx[:-1], switch_idx[1:]):
                # plot bar from last change to current one with specific color
                ax.barh(
                    name,
                    width=time[end_idx] - time[start_idx],
                    left=time[start_idx],
                    color=color_mapping[values[end_idx]],
                )
            ax.set_xlabel("Time [s]")
        if legend:
            ax.legend(handles=[mpatches.Patch(color=color, label=value) for value, color in color_mapping.items()])
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
        _, x_values, y_values = get_xy_from_timeseries(self[:, x_signal_name],
                                                       self[:, y_signal_name])
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
            names = list(self.signals)
        df = pd.DataFrame()
        for name in names:
            try:
                # Create a new DataFrame which will be concatenated to
                # the other DataFrame. Before that this DataFrame will be
                # prepared.
                df_concat = self[:, name].reset_index()
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
            names = list(self.signals)
        if timestamps is None:
            # get the time span of data
            # important: smallest time span of all signals is used
            time_min, time_max = 0, np.infty
            for name in names:
                time_min_local = self[:, name].index[0]
                time_max_local = self[:, name].index[-1]
                if time_min_local > time_min:
                    time_min = time_min_local
                if time_max_local < time_max:
                    time_max = time_max_local

            # set global variables
            length = int((time_max - time_min) * frequency)
            timestamps = np.linspace(time_min, time_max, length)

        dataframe = pd.DataFrame(timestamps, columns=["time"])
        for name in names:
            time, values = self[:, name].index, self[:, name].values
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

    def to_parquet(self, path):
        path = Path(path)
        os.mkdir(path)
        for message_name in self.messages:
            self[message_name].to_parquet(path / (message_name + ".parquet"))

    def to_sql(self, conn):
        for message_name in self.messages:
            self[message_name].to_sql(message_name, conn)

    def to_mat(self, path):
        path = Path(path)
        signal_data = {}
        for message_name in self.messages:
            for signal_name in self[message_name].columns:
                signal_data[signal_name] = np.array([
                    self[message_name, signal_name].index,
                    self[message_name, signal_name].values
                ]).T

        savemat(path, signal_data)


def from_file(dbc_db, path, names=None, always_convert=False,
              verbose=True, **kwargs):
    """
    Create CANDataLog object from a file, which can be a directory
    with .parquet files, a .blf (raw binary) or .mat (converted) file.

    If the data is already converted in the .parquet or mat .format,
    it preferably loaded from those in that order.

    Parameters
    ----------
    dbc_db : cantools.db.Database
        The dbc database which was used to convert the data from a binary
        format.
    path : str, path object
        absolute or relative path of the log file with or without the extension.
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
    path = Path(path).with_suffix("")
    if os.path.isdir(path) and not always_convert:
        if verbose:
            print("Loading from parquet...")
        return from_parquet(dbc_db, path)
    elif os.path.isfile(path.with_suffix(".mat")) and not always_convert:
        if verbose:
            print("Loading from mat-file...")
        return from_mat(dbc_db, path.with_suffix(".mat"))
    else:
        if verbose:
            print(
                "Converting data to readable format... "
                "this might take several minutes"
            )
        return from_blf(dbc_db, path.with_suffix(".blf"), names)


def from_parquet(dbc_db, path, **kwargs):
    path = Path(path).with_suffix("")
    log_data = dict()
    for file_path in path.with_suffix("").glob("*.parquet"):
        log_data[file_path.stem] = pd.read_parquet(file_path)
    return CANDataLog(log_data, dbc_db, **kwargs)


def from_sql(dbc_db, conn, table_names, **kwargs):
    log_data = dict()
    for table_name in table_names:
        # SQLite DBAPI connection mode not supported for read_sql_table
        log_data[table_name] = pd.read_sql_query(
            f"SELECT * FROM {table_name}",
            conn,
            index_col="timestamp",
        )
    return CANDataLog(log_data, dbc_db, **kwargs)


def from_mat(dbc_db, path, **kwargs):
    path = Path(path)
    signal_data = loadmat(path.with_suffix(".mat"))
    dfs = dict()
    for message in dbc_db.messages:
        message_data = list()
        signal_names = list()
        try:
            for signal in message.signals:
                message_data.append(signal_data[signal.name][:, 1])
                signal_names.append(signal.name)
            dfs[message.name] = pd.DataFrame(
                # For string dtypes the time values are objects
                index=signal_data[signal.name][:, 0].astype(np.float64),
                data=np.array(message_data).T,
                columns=signal_names,
            )
        except KeyError:
            pass

    return CANDataLog(dfs, dbc_db, **kwargs)


def from_blf(dbc_db, path, names=None, **kwargs):
    path = Path(path)
    log_data = can.BLFReader(path)
    log_data = decode_data(log_data, dbc_db)

    if names is None:
        ret_value = log_data
    else:
        ret_value = defaultdict(dict)
        for name in names:
            message_name, signal_name = name
            ret_value[message_name][signal_name] = log_data[message_name][signal_name]

    return CANDataLog(ret_value, dbc_db, **kwargs)


def from_fake(dbc_db, messages_properties, **kwargs):
    """
    Create a data log with propterties given in a list of dicts with key name
    arguments of create_fake_can_data function.

    Parameters
    ----------
    dbc_db : cantools.db.Database
        The dbc database which was used to convert the data from a binary
        format.
    messages_properties : :obj:`dict`
        Key-Value pairs of the properties. Signal properties must be stored under "signals" key. See create_fake_can_data for more
        information.

    Returns
    -------
    CANDataLog
        Class of data log.

    """
    log_data = {}
    messages_properties = copy.deepcopy(messages_properties)
    for message_properties in messages_properties:
        message_name = message_properties.pop("name")
        signals_properties = message_properties.pop("signals")
        time = create_fake_can_time(**message_properties)
        data = dict()
        for signal_properties in signals_properties:
            signal_name = signal_properties.pop("name")
            data[signal_name] = create_fake_can_data(time, **signal_properties)
        log_data[message_name] = pd.DataFrame(index=time, data=data)

    return CANDataLog(log_data, dbc_db, **kwargs)


def load_dbc(path, verbose=True):
    """
    Load all dbc files from specified folder add to dbc database.

    Parameters
    ----------
    path : str
        Absolute or relative path, which contains dbc files.
        Glob string to match dbc files. List of paths to dbc files.
        The extension is not being checked.
    verbose : bool, optional
        Set to False to have no readout. The default is True.

    Returns
    -------
    dbc_db : cantools.db.Database
        dbc database to convert the data from binary format.

    """
    if isinstance(path, str):
        if Path(path).is_dir():
            path = os.path.join(path, "*.dbc")
        dbc_paths = glob.glob(path)
        assert len(dbc_paths) > 0, f"'{path}' did not match any file"
    elif isinstance(path, Iterable):
        dbc_paths = path
    # Could be path object
    else:
        dbc_paths = [path]
    assert len(dbc_paths) > 0, f"No dbc-files path '{path}'!"
    dbc_db = cantools.db.Database(
        messages=None, nodes=None, buses=None, version=None
    )
    if verbose:
        print(f"Loading files: {dbc_paths}")
    for dbc_path in dbc_paths:
        dbc_db.add_dbc_file(dbc_path)
    if verbose:
        print("Finished loading.")
    return dbc_db

def get_dtypes(dbc_db):
    dtypes = defaultdict(dict)
    for message in dbc_db.messages:
        for signal in message.signals:
            if signal.choices is not None:
                dtypes_of_choices = tuple(set([type(value.value) for key, value in signal.choices.items()]))
                if len(dtypes_of_choices) != 1:
                    raise ValueError(f"Mixed dtypes in choices is not allowed ({message.name}, {signal.name})")
                value = tuple(signal.choices.values())[0].name
                if isinstance(value, str):
                    dtype_of_choices = "string"
                else:
                    raise ValueError(f"Dtype {type(value)} for choices is not supported in message ({message.name}, {signal.name})")
            else:
                dtype_of_choices = None

            if dtype_of_choices == "string":
                dtype = "string"
            else:
                if signal.length == 1:
                    dtype = "boolean"
                elif isinstance(signal.scale, int) and isinstance(signal.offset, int):
                    dtype = "Int64"
                elif isinstance(signal.scale, float) or isinstance(signal.offset, float):
                    dtype = "Float64"
                else:
                    raise ValueError(f"Could not detect data type ({message.name}, {signal.name})")
            dtypes[message.name][signal.name] = dtype
    return dict(dtypes)


def decode(dbc_db, message):
    try:
        arbitration_id, timestamp, data = message
        return arbitration_id, dbc_db.decode_message(arbitration_id, data), timestamp
    except:
        return arbitration_id, None, None


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
    messages = [(msg.arbitration_id, msg.timestamp, tuple(msg.data)) for msg in log]
    messages_decoded =  [decode(dbc_db, message) for message in messages]
    messages_grouped = defaultdict(list)
    error_ids = set()
    for arbitration_id, data, timestamp in messages_decoded:
        if data:
            name = dbc_db.get_message_by_frame_id(arbitration_id).name
            messages_grouped[name].append({**{"timestamp": timestamp}, **data})
        else:
            error_ids.add(arbitration_id)
    dfs = dict()
    for name in messages_grouped.keys():
        dfs[name] = pd.DataFrame(messages_grouped[name]).set_index("timestamp")
    if error_ids:
        print("The following IDs caused errors: " + str(error_ids))

    dtypes = get_dtypes(dbc_db)
    for message_name in dfs.keys():
        dfs[message_name] = dfs[message_name].astype(dtypes[message_name])
    return dfs


def get_xy_from_timeseries(x_data, y_data):
    """
    Return the x, y values of time series as pairs.

    Parameters
    ----------
    x_data : pd.Series
    y_data : pd.Series

    Returns
    -------
    :obj:`tuple` of numpy.ndarray
        Tuple which contains the unified timestamp, one original value signal
        and one interpolated value signal.

    """
    t_x, data_x = x_data.index, x_data.values
    t_y, data_y = y_data.index, y_data.values

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
                log_data[:, key].index,
                log_data[:, key].values,
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
    x_data, y_data = signal.index, signal.values
    start = np.abs(np.array(x_data) - start).argmin()
    end = np.abs(np.array(x_data) - end).argmin()

    return x_data, y_data, start, end


def create_fake_can_time(start, stop, period, time_noise=0, verbose=False):
    """
    Create fake CAN time for testing.

    Parameters
    ----------
    start : float
        Start timestamp.
    stop : float
        End timestamp.
    period : float, optional
        Time period between signals. The default is 0.01.
    time_noise : float, optional
        Stanard deviation of the jitter of the time period between signals.
        The default is 0.05.
    """

    if verbose:
        if time_noise > 0.1:
            print("Time noise is pretty large (Over 10% of period)")

    steps = int((stop - start) / period)
    time = np.linspace(start, stop, steps)

    # adding noise to time
    time += np.random.normal(scale=time_noise * period, size=len(time))

    assert np.diff(time).min() > 0

    return time


def create_fake_can_data(
        time,
        kind,
        amplitude=1,
        offset=0,
        signal_noise=0.02,
        signal_type="sin",
        verbose=True,
    ):
    """
    Create fake CAN data for testing.

    This function uses sane default values to make usage easier.

    Parameters
    ----------
    time: np.array
        Time signal.
    kind: str
        Kind of data to create. Can either be "float" or "categorical".
    amplitude : float, optional
        Amplitude of the value signal. The default is 1.
    offset : float, optional
        Offset of the value signal. The default is 0.
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
    ValueError
        When a not supported signal kind is used.

    Returns
    -------
    data : numpy.ndarray
        Array of time and value data.

    """
    if verbose:
        if signal_noise > 0.05:
            print("Signal noise is pretty large (Over 5% of amplitude)")

    if kind == "float":
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

    elif kind == "categorical":
        signal = np.random.choice([0, 1], size=len(time), p=[0.1, 0.9])
    else:
        raise ValueError("The signal kind '{}' is "
                        "not supported.".format(kind))

    return signal
