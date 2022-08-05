# Databases and SQL for Data Science with Python <!-- omit in toc -->

- [Basic SQL](#basic-sql)
- [Relational Database](#relational-database)
- [Refine your results](#refine-your-results)
- [Functions, Multiple Tables, and Sub-queries](#functions-multiple-tables-and-sub-queries)
  - [Built-in Database Functions](#built-in-database-functions)
    - [Aggregate or Column Functions](#aggregate-or-column-functions)
    - [Scalar and string functions](#scalar-and-string-functions)
    - [Date and time built-in functions](#date-and-time-built-in-functions)
  - [Sub-Queries and Nested Selects](#sub-queries-and-nested-selects)
  - [Multiple Tables](#multiple-tables)
- [Accessing databases using Python](#accessing-databases-using-python)
## Basic SQL

What is SQL?
- A language used for relational database
- Query data

What is data?
- Facts (words, numbers)
- Pictures
- One of the most critical assets of any business

What is a database?
- A repository of data
- Provides the functionality for adding, modifying and querying that data
- Different kinds of databases store data in different forms

Relational Database
- Data stored in tabular form - columns and rows
- Columns contain item properties e.g. Last Name, First Name, etc.
- Table is collection of related things e.g. Employees, Authors, etc.
- Relationships can exist between tables (hence: "relational")

DBMS
- Database: repository of data
- DBMS: Database Management System - software to manage databases
- Database, Database Server, Database System, Data Server, DBMS - often used interchangeably

RDBMS
- RDBMS - Relational database management system
- A set of software tools that controls the data
  - access, organization, and storage
- Examples are: MySQL, Oracle Database, IBM Db2, etc.

Basic SQL Commands
- Create a table
- Insert
- Select
- Update
- Delete


## Relational Database

Relational Model
- Most used data model
- Allows for data independence
  - logical data independence
  - physical data independence
  - physical storage independence
- Data is stored in tables

Entity-Relationship Model
- Used as a tool to design relational databases
  
Mapping Entity Diagrams to Tables
- Entities become tables
- Attributes get translated into columns


<img src="res/cloud-db.png" width="700"></img>

Examples of Cloud databases
- IBM Db2
- Databases for PostgreSQL
- Oracle Database Cloud Service
- Microsoft Azure SQL Database
- Amazon Relational Database Services (RDS)

Available as:
- VMs or Managed Service
- Single or Multi-tenant

<img src="res/db-service.png" width="500"></img>


SQL Statement types
- DDL (Data Definition Language) statements:
  - define, change, or drop data
  - common DDL:
    - create
    - alter
    - truncate
    - drop
- DML (Data Manipulation Language) statements:
  - read and modify data
  - CRUD operations (Create, Read, Update & Delete rows)
  - common DML:
    - insert
    - select
    - update
    - delete

## Refine your results

- `distinct` clause
- `group by` clause
- `having` clause

```sql
--Query 1.1--

SELECT F_NAME , L_NAME
FROM EMPLOYEES
WHERE ADDRESS LIKE '%Elgin,IL%';

--Query 1.2--

SELECT F_NAME , L_NAME
FROM EMPLOYEES
WHERE B_DATE LIKE '197%';

--Query 1.3--

SELECT *
FROM EMPLOYEES
WHERE (SALARY BETWEEN 60000 AND 70000) AND DEP_ID = 5;

--Query 2.1--

SELECT F_NAME, L_NAME, DEP_ID 
FROM EMPLOYEES
ORDER BY DEP_ID;

--Query 2.2--

SELECT F_NAME, L_NAME, DEP_ID 
FROM EMPLOYEES
ORDER BY DEP_ID DESC, L_NAME DESC;

--Optional Query 2.3--

SELECT D.DEP_NAME , E.F_NAME, E.L_NAME
FROM EMPLOYEES as E, DEPARTMENTS as D
WHERE E.DEP_ID = D.DEPT_ID_DEP
ORDER BY D.DEP_NAME, E.L_NAME DESC;

--Query 3.1--

SELECT DEP_ID, COUNT(*)
FROM EMPLOYEES
GROUP BY DEP_ID;

--Query 3.2--

SELECT DEP_ID, COUNT(*), AVG(SALARY)
FROM EMPLOYEES
GROUP BY DEP_ID;

--Query 3.3--

SELECT DEP_ID, COUNT(*) AS "NUM_EMPLOYEES", AVG(SALARY) AS "AVG_SALARY"
FROM EMPLOYEES
GROUP BY DEP_ID;

--Query 3.4--

SELECT DEP_ID, COUNT(*) AS "NUM_EMPLOYEES", AVG(SALARY) AS "AVG_SALARY"
FROM EMPLOYEES
GROUP BY DEP_ID
ORDER BY AVG_SALARY;

--Query 3.5--

SELECT DEP_ID, COUNT(*) AS "NUM_EMPLOYEES", AVG(SALARY) AS "AVG_SALARY"
FROM EMPLOYEES
GROUP BY DEP_ID
HAVING count(*) < 4
ORDER BY AVG_SALARY;

```

## Functions, Multiple Tables, and Sub-queries

### Built-in Database Functions

Built-in Functions
- Most databases come with built-in SQL functions
- Built-in functions can be included as part of SQL statements
- Database functions can significantly reduce the amount of data that needs to be retrieved
- Can speed up data processing

#### Aggregate or Column Functions

- input: Collection of values (e.g. entire column)
- output: Single value
- Examples:
  - SUM()
  - MIN()
  - MAX()
  - AVG()

#### Scalar and string functions

- scalar: perform operations on every input value
- examples:
  - ROUND()
  - LENGTH()
  - UCASE()
  - LCASE()

```sql
select DISTINCT(UCASE(ANIMAL)) FROM PETRESCUE
```

#### Date and time built-in functions

```sql
--Query C1: Enter a function that displays the day of the month when cats have been rescued.
select DAY(RESCUEDATE) from PETRESCUE where ANIMAL = 'Cat';

--Query C2: Enter a function that displays the number of rescues on the 5th month.
select SUM(QUANTITY) from PETRESCUE where MONTH(RESCUEDATE)='05';

--Query C3: Enter a function that displays the number of rescues on the 14th day of the month.
select SUM(QUANTITY) from PETRESCUE where DAY(RESCUEDATE)='14';

--Query C4: Animals rescued should see the vet within three days of arrivals. 
--Enter a function that displays the third day from each rescue.
select (RESCUEDATE + 3 DAYS) from PETRESCUE;

--Query C5: Enter a function that displays the length of time the animals have been rescued; 
--the difference between today’s date and the recue date.
select (CURRENT DATE - RESCUEDATE) from PETRESCUE;
```

### Sub-Queries and Nested Selects



- how sub-queries and nested queries can be used to form richer queries
- how they can overcome some of the limitations of aggregate functions
- use sub-queries in 
  - `WHERE` clause
  - list of `columns`
  - `FROM` clause


One of the limitations of built in aggregate functions, like the average function, is that they cannot always be evaluated in the WHERE clause.
```sql
Select * from employees, where salary > AVG(salary)
```

should use **sub-select** expression:

```sql
Select EMP_ID, F_NAME, L_NAME, SALARY from employees where SALARY < ( select AVG (SALARY) from employees)
```

Use the average function in a sub-query placed in the list of the columns:

```sql
select EMP_ID, SALARY, (select AVG(SALARY) from employees ) AS AVG_SALARY from employees
```

Make the sub-query be part of the FROM clause:

```sql
Select * from (select EMP_ID, F_NAME, L_NAME, DEP_ID from employees) AS EMP4ALL
```


### Multiple Tables

Accessing Multiple Tables with Sub-Queries

```sql
--- Query 1A ---
select * from employees where JOB_ID IN (select JOB_IDENT from jobs)
;	
--- Query 1B ---	
select * from employees where JOB_ID IN (select JOB_IDENT from jobs where JOB_TITLE= 'Jr. Designer')
;
--- Query 1C ---
select JOB_TITLE, MIN_SALARY,MAX_SALARY,JOB_IDENT from jobs where JOB_IDENT IN (select JOB_ID from employees where SALARY > 70000 )
;	
--- Query 1D ---
select JOB_TITLE, MIN_SALARY,MAX_SALARY,JOB_IDENT from jobs where JOB_IDENT IN (select JOB_ID from employees where YEAR(B_DATE)>1976 )
;
--- Query 1E ---
select JOB_TITLE, MIN_SALARY,MAX_SALARY,JOB_IDENT from jobs where JOB_IDENT IN (select JOB_ID from employees where YEAR(B_DATE)>1976 and SEX='F' )
;
```

Accessing Multiple Tables with Implicit Joins

```sql
--- Query 2A ---
select * from employees, jobs
;
--- Query 2B ---
select * from employees, jobs where employees.JOB_ID = jobs.JOB_IDENT
;
--- Query 2C ---
select * from employees E, jobs J where E.JOB_ID = J.JOB_IDENT
;
--- Query 2D ---
select EMP_ID,F_NAME,L_NAME, JOB_TITLE from employees E, jobs J where E.JOB_ID = J.JOB_IDENT
;
--- Query 2E ---
select E.EMP_ID,E.F_NAME,E.L_NAME, J.JOB_TITLE from employees E, jobs J where E.JOB_ID = J.JOB_IDENT
;
```

## Accessing databases using Python



<img src="res/sql-api.png" width="500"></img>


<img src="res/db-APIs.png" width="500"></img>

Concepts of the Python DB API

- Connection Objects
  - Database connections
  - Manage transactions
- Cursor Objects
  - Database Queries
  - Scroll through result set
  - Retrieve results

connect methods:
- `.cursor()`
- `.commit()`
- `.rollback()`
- `.close()`

cursor methods:
- `.callproc()`
- `.execute()`
- `.executemany()`
- `.fetchone()`
- `.fetchmany()`
- `.fetchall()`
- `.nextset()`
- `.arraysize()`
- `.close()`



<img src="res/cursor.png" width="400"></img>



<img src="res/db-api-py.png" width="400"></img>
