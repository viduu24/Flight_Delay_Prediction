CREATE DATABASE airline_project;
USE airline_project;


CREATE TABLE flights (
    year INT,
    month INT,
    day_of_month INT,
    day_of_week INT,
    fl_date DATE,
    op_unique_carrier VARCHAR(10),
    op_carrier_fl_num INT,
    origin VARCHAR(10),
    origin_city_name VARCHAR(100),
    origin_state_nm VARCHAR(50),
    dest VARCHAR(10),
    dest_city_name VARCHAR(100),
    dest_state_nm VARCHAR(50),
    crs_dep_time INT,
    dep_time FLOAT,
    dep_delay FLOAT,
    taxi_out FLOAT,
    wheels_off FLOAT,
    wheels_on FLOAT,
    taxi_in FLOAT,
    crs_arr_time INT,
    arr_time FLOAT,
    arr_delay FLOAT,
    cancelled INT,
    cancellation_code VARCHAR(5),
    diverted INT,
    crs_elapsed_time FLOAT,
    actual_elapsed_time FLOAT,
    air_time FLOAT,
    distance FLOAT,
    carrier_delay FLOAT,
    weather_delay FLOAT,
    nas_delay FLOAT,
    security_delay FLOAT,
    late_aircraft_delay FLOAT
);

-- LOAD DATA LOCAL INFILE '/Users/ahmadjabbar/Desktop/S-MSU/Spring 2026/811/Project/Flight_Data/flight_data_2024_sample.csv'
-- INTO TABLE flights
-- FIELDS TERMINATED BY ','
-- ENCLOSED BY '"'
-- LINES TERMINATED BY '\n'
-- IGNORE 1 ROWS;

-- SHOW VARIABLES LIKE 'local_infile';
-- SET GLOBAL local_infile = 1;



SELECT * FROM flights
LIMIT 10