import os

import numpy as np
import pandas as pd
import pooch

from relbench.data import Database, RelBenchDataset, Table
from relbench.tasks.f1 import DidNotFinishTask, PositionTask, QualifyingTask
from relbench.utils import unzip_processor


class F1Dataset(RelBenchDataset):
    name = "rel-f1"
    val_timestamp = pd.Timestamp("2005-01-01")
    test_timestamp = pd.Timestamp("2010-01-01")
    max_eval_time_frames = 40
    task_cls_list = [
        PositionTask,
        DidNotFinishTask,
        QualifyingTask,
    ]

    def __init__(
        self,
        *,
        process: bool = False,
        cache_dir: str = None,
    ):
        self.cache_dir = cache_dir
        self.name = f"{self.name}"
        super().__init__(process=process)

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        url = "https://relbench.stanford.edu/data/relbench-f1-raw.zip"

        path = pooch.retrieve(
            url,
            known_hash="2933348953b30aa9723b4831fea8071b336b74977bbcf1fb059da63a04f06eba",
            progressbar=True,
            processor=unzip_processor,
            path=self.cache_dir,
        )

        path = os.path.join(path, "raw")

        print("Current working directory:", os.getcwd())

        circuits = pd.read_csv(os.path.join(path, "circuits.csv"))
        drivers = pd.read_csv(os.path.join(path, "drivers.csv"))
        results = pd.read_csv(os.path.join(path, "results.csv"))
        races = pd.read_csv(os.path.join(path, "races.csv"))
        standings = pd.read_csv(os.path.join(path, "driver_standings.csv"))
        constructors = pd.read_csv(os.path.join(path, "constructors.csv"))
        constructor_results = pd.read_csv(os.path.join(path, "constructor_results.csv"))
        constructor_standings = pd.read_csv(
            os.path.join(path, "constructor_standings.csv")
        )
        qualifying = pd.read_csv(os.path.join(path, "qualifying.csv"))

        ## remove columns that are irrelevant, leak time, or have too many missing values
        races.drop(
            columns=[
                "url",
                "fp1_date",
                "fp1_time",
                "fp2_date",
                "fp2_time",
                "fp3_date",
                "fp3_time",
                "quali_date",
                "quali_time",
                "sprint_date",
                "sprint_time",
            ],
            inplace=True,
        )

        circuits.drop(
            columns=["url", "alt"],
            inplace=True,
        )

        drivers.drop(
            columns=["number", "url"],
            inplace=True,
        )

        results.drop(
            columns=[
                "positionText",
                "time",
                "fastestLapTime",
                "fastestLapSpeed",
            ],
            inplace=True,
        )

        standings.drop(
            columns=["positionText"],
            inplace=True,
        )

        constructors.drop(
            columns=["url"],
            inplace=True,
        )

        constructor_standings.drop(
            columns=["positionText"],
            inplace=True,
        )

        qualifying.drop(
            columns=["q1", "q2", "q3"],
            inplace=True,
        )

        ## replase missing data and combine date and time columns
        races["time"] = races["time"].replace(r"^\\N$", "00:00:00", regex=True)
        races["date"] = races["date"] + " " + races["time"]
        ## change time column to unix time
        races["date"] = pd.to_datetime(races["date"])

        # add time column to other tables
        results = results.merge(races[["raceId", "date"]], on="raceId", how="left")
        standings = standings.merge(races[["raceId", "date"]], on="raceId", how="left")
        constructor_results = constructor_results.merge(
            races[["raceId", "date"]], on="raceId", how="left"
        )
        constructor_standings = constructor_standings.merge(
            races[["raceId", "date"]], on="raceId", how="left"
        )

        qualifying = qualifying.merge(
            races[["raceId", "date"]], on="raceId", how="left"
        )

        # subtract a day from the date to account for the fact
        # that the qualifying time is the day before the main race
        qualifying["date"] = qualifying["date"] - pd.Timedelta(days=1)

        # replace "\N" with NaN in all tables
        results = results.replace(r"^\\N$", np.nan, regex=True)

        # Convert non-numeric values to NaN in the specified column
        results["rank"] = pd.to_numeric(results["rank"], errors="coerce")
        results["number"] = pd.to_numeric(results["number"], errors="coerce")
        results["grid"] = pd.to_numeric(results["grid"], errors="coerce")
        results["position"] = pd.to_numeric(results["position"], errors="coerce")
        results["points"] = pd.to_numeric(results["points"], errors="coerce")
        results["laps"] = pd.to_numeric(results["laps"], errors="coerce")
        results["milliseconds"] = pd.to_numeric(
            results["milliseconds"], errors="coerce"
        )
        results["fastestLap"] = pd.to_numeric(results["fastestLap"], errors="coerce")

        tables = {}

        tables["races"] = Table(
            df=pd.DataFrame(races),
            fkey_col_to_pkey_table={
                "circuitId": "circuits",
            },
            pkey_col="raceId",
            time_col="date",
        )

        tables["circuits"] = Table(
            df=pd.DataFrame(circuits),
            fkey_col_to_pkey_table={},
            pkey_col="circuitId",
            time_col=None,
        )

        tables["drivers"] = Table(
            df=pd.DataFrame(drivers),
            fkey_col_to_pkey_table={},
            pkey_col="driverId",
            time_col=None,
        )

        tables["results"] = Table(
            df=pd.DataFrame(results),
            fkey_col_to_pkey_table={
                "raceId": "races",
                "driverId": "drivers",
                "constructorId": "constructors",
            },
            pkey_col="resultId",
            time_col="date",
        )

        tables["standings"] = Table(
            df=pd.DataFrame(standings),
            fkey_col_to_pkey_table={"raceId": "races", "driverId": "drivers"},
            pkey_col="driverStandingsId",
            time_col="date",
        )

        tables["constructors"] = Table(
            df=pd.DataFrame(constructors),
            fkey_col_to_pkey_table={},
            pkey_col="constructorId",
            time_col=None,
        )

        tables["constructor_results"] = Table(
            df=pd.DataFrame(constructor_results),
            fkey_col_to_pkey_table={"raceId": "races", "constructorId": "constructors"},
            pkey_col="constructorResultsId",
            time_col="date",
        )

        tables["constructor_standings"] = Table(
            df=pd.DataFrame(constructor_standings),
            fkey_col_to_pkey_table={"raceId": "races", "constructorId": "constructors"},
            pkey_col="constructorStandingsId",
            time_col="date",
        )

        tables["qualifying"] = Table(
            df=pd.DataFrame(qualifying),
            fkey_col_to_pkey_table={
                "raceId": "races",
                "driverId": "drivers",
                "constructorId": "constructors",
            },
            pkey_col="qualifyId",
            time_col="date",
        )

        return Database(tables)
