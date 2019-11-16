#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MetaData for he CANDataLog class.
"""
import datetime

class MetaData(dict):
    """
    This holds all metadata information of CANDataLog.

    Only defined metadata keys can be set. Depending on the key checks are
    being conducted to ensure good data quality.

    """
    def __setitem__(self, key, value):
        """
        Set item in dictionary and check data quality.

        Parameters
        ----------
        key : str
           Key value of the metadata attribute.
        value : any
            Value of the metadata attribute. This will be checked for data
            quality.

        Returns
        -------
        None.

        """
        assert isinstance(key, str)
        # Get checking function and execute with value
        getattr(self, key, key_not_found)(key, value)
        super(MetaData, self).__setitem__(key, value)

    # pylint: disable=R0201
    def file_hash_blf(self, key, _):
        """
        File hash of the original .blf file.

        """
        not_changeable(key)

    # pylint: disable=R0201
    def file_hash_mat(self, key, _):
        """
        File hash of the converted .mat file.

        """
        not_changeable(key)

    # pylint: disable=R0201
    def source(self, key, _):
        """
        The source from where this object was created.

        It can be either from a file, database or from a fake data generator.

        """
        not_changeable(key)

    # pylint: disable=R0201
    def name(self, key, value):
        """
         Name of the data log.

         """
        must_be_instance_of(key, value, str)
        must_be_less_than_chars(key, value, 30)

    # pylint: disable=R0201
    def dbc_commit_hash(self, key, value):
        """
        Commit hash of the dbc, if its versioned wih git, which it should.

        It should be the full commit hash not the shortened one.
        """
        must_be_instance_of(key, value, str)
        must_be_equal_chars(key, value, 40)

    # pylint: disable=R0201
    def date(self, key, value):
        """
        The date on which the log was recorded.

        """
        must_be_instance_of(key, value, datetime.date)

    # pylint: disable=R0201
    def start_time(self, key, value):
        """
        Start time of the log.

        This parameter is optional in the class as it is sometime hard to get
        the specific time from the log files.

        Together with the date it marks one point in time when the session took
        place. Both values are separated so the time value can be let empty and
        be marked as such.

        """
        must_be_instance_of(key, value, datetime.time)

    # pylint: disable=R0201
    def description(self, key, value):
        """
        Short description of the log. This can be observation at the track or
        vehicle or other important to remember circumstances.

        """
        must_be_instance_of(key, value, str)
        must_be_less_than_chars(key, value, 140)

    # pylint: disable=R0201
    def event(self, key, value):
        """
        Event of the log.

        For example FSG, FSN, VDE Race or Conti After Race. Names dont have to
        be abreviated and no validation takes place.

        """
        must_be_instance_of(key, value, str)
        must_be_less_than_chars(key, value, 30)

    # pylint: disable=R0201
    def location(self, key, value):
        """
        Location of the log.

        This value can be used to name the city event or testing took place.
        Its also possible to put GPS coordinates in this field.
        """
        must_be_instance_of(key, value, str)
        must_be_less_than_chars(key, value, 30)

    # pylint: disable=R0201
    def accumulator_temperature_name(self, key, value):
        """
        Starting string of the temperature sensor signals.

        """
        must_be_instance_of(key, value, str)

    # pylint: disable=R0201
    def temperature_sensors_per_stack(self, key, value):
        """
        Number of temperature sensors per stack.

        """
        must_be_instance_of(key, value, int)

    # pylint: disable=R0201
    def number_of_stacks(self, key, value):
        """
        Number of stacks in the accumulator.


        """
        must_be_instance_of(key, value, int)


def key_not_found(key, _):
    """
    Return ValueError if key is not defined in MetaData class.

    Parameters
    ----------
    key: str
        Key, which was not found.

    Returns
    -------
    None.

    """
    raise ValueError(f"The metadata key '{key}' does not exist!")

def not_changeable(key):
    """
    Return ValueError, because this metadata attribute is not changeable.

    Parameters
    ----------
    key: str
        Key of the metadata attribute which is not changeable.

    Returns
    -------
    None.

    """

    raise ValueError(f"'{key}' cannot be changed")

def must_be_instance_of(key, value, instance):
    """
    Check whether value is instance of specified type.

    Parameters
    ----------
    key : str
        Key of the metadata attribute which is being checked.
    value : any
        Value of the metadata attribute which is being checked.
    instance : any
        Instance of the type which the value should have.

    Returns
    -------
    None.

    """
    assert isinstance(value, instance), (f"'{key}' must be instance of "
                                         f"{instance.__name__}")

def must_be_less_than_chars(key, value, length):
    """
    Check whether value is less than given amount of characters.

    Parameters
    ----------
    key : str
        Key of the metadata attribute which is being checked.
    value : any
        Value of the metadata attribute which is being checked.
    length : int
        Length in characters.

    Returns
    -------
    None.

    """
    assert len(value) < length, (f"'{key}' must be less than "
                                 f"{length} characters")

def must_be_equal_chars(key, value, length):
    """
    Check whether value is exact given amount of characters.

    Parameters
    ----------
    key : str
        Key of the metadata attribute which is being checked.
    value : any
        Value of the metadata attribute which is being checked.
    length : int
        Length in characters.

    Returns
    -------
    None.

    """
    assert len(value) == length, (f"'{key}' must be equal to "
                                  f"{length} characters")
