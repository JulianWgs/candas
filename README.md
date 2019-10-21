# CANdas - The dataframe for CAN* bus data

> Manage and analyze CAN bus data elegantly and efficiently, locally or in the cloud.

![CANdas Jupyter Lab example](http://img.lionsracing.de/1O1PIJXVSJ4ZCLSM.png)

*[Controller Area Network](https://en.wikipedia.org/wiki/CAN_bus)

## Features

- Common format for dealing with CAN data
- Enrich plots of the logging data with data from the dbc files automatically
- Versatile and extensible plotting functions for all kinds of signals
- Easily export CAN data to a pandas dataframe
- Data can be pushed to a SQL database
- Download logging file from SQL database instead of having all of them on disk
- Method chaining philosophy to create powerful and minimalistic pipelines

## Documentation

Extensive documentation can be found here: https://lionsracing.gitlab.io/candas/index.html

## Installation

```bash
git clone https://gitlab.com/lionsracing/candas.git
pip install -e candas
```

## Contributing

1. Fork the repository.

2. Install prerequisites.

   ```
   pip install -r requirements.txt
   ```

3. Implement the new feature or bug fix.

4. Implement test case(s) to ensure that future changes do not break legacy.

5. Run the tests.

6. Create a pull request.