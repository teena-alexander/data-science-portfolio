

--Given the Tb_Transactions_Cube stored above solve the following queries:
--1) Aggregates by combinations of supplier name and product name?
SELECT
*
FROM Tb_Transactions_Cube
WHERE "Supplier Name" IS NOT NULL
AND "Supplier City" IS NOT NULL
AND "Supplier State" IS NOT NULL
AND "Consumer Name" IS NULL
AND "Consumer City" IS NULL
AND "Consumer State" IS NULL
AND "Product Name" IS NOT NULL ;

--2) Aggregates by supplier states?
SELECT
*
FROM Tb_Transactions_Cube
WHERE "Supplier Name" IS NULL
AND "Supplier City" IS NULL
AND "Supplier State" IS NOT NULL
AND "Consumer Name" IS NULL
AND "Consumer City" IS NULL
AND "Consumer State" IS NULL
AND "Product Name" IS  NULL ;

--3) Number of transactions by supplier-city and consumer-city combinations?
SELECT
"Supplier City",
"Consumer City",
[Number of Transactions]
FROM Tb_Transactions_Cube
WHERE "Supplier Name" IS NULL
AND "Supplier City" IS NOT NULL
AND "Supplier State" IS NOT NULL
AND "Consumer Name" IS NULL
AND "Consumer City" IS NOT NULL
AND "Consumer State" IS NOT NULL
AND "Product Name" IS  NULL ;

--4) Name of each product and quantity Wisconsin suppliers sold of that
--product?
SELECT [Product Name],
SUM([Total Transactions Quantity]) AS [Total Quantity]
FROM Tb_Transactions_Cube
WHERE 
[Supplier Name] IS NOT NULL 
AND[Supplier City] IS NOT NULL
AND[Supplier State]='Wisconsin'
AND[Consumer Name]IS NULL
AND[Consumer City]IS NULL
AND[Consumer State]IS NULL
AND[Product Name] IS NOT NULL
GROUP BY [Product Name];


--5) Quantity of sales aggregated by product and supplier state
SELECT "Product Name", 
"Supplier State" ,
[Total Transactions Quantity]
FROM Tb_Transactions_Cube
WHERE "Supplier Name" IS NULL
AND "Supplier City" IS NULL
AND "Supplier State" IS NOT NULL
AND "Consumer Name" IS NULL
AND "Consumer City" IS NULL
AND "Consumer State" IS NULL
AND "Product Name" IS NOT NULL;


--6) Quantity of computer sales aggregated by suppliers in Wisconsin?
SELECT [Supplier Name],
SUM([Total Transactions Quantity]) AS [Total Quantity]
FROM Tb_Transactions_Cube
WHERE 
[Supplier Name] IS NOT NULL 
AND[Supplier City] IS NOT NULL
AND[Supplier State]='Wisconsin'
AND[Consumer Name]IS NULL
AND[Consumer City]IS NULL
AND[Consumer State]IS NULL
AND[Product Name] = 'Computer'
GROUP BY [Supplier Name];



--7) Quantity of auto sales by each supplier from Wisconsin to each auto
--consumer in Illinois?
SELECT 
"Supplier Name",
"Consumer Name",
[Total Transactions Quantity]
FROM Tb_Transactions_Cube
WHERE "Supplier Name" IS NOT NULL
AND "Supplier City" IS NOT NULL
AND "Supplier State" = 'Wisconsin'
AND "Consumer Name" IS NOT NULL
AND "Consumer City" IS NOT NULL
AND "Consumer State" ='Illinois'
AND "Product Name" ='Auto';


--8) Quantity of each product sold by each supplier in Madison to each
--consumer in Illinois?
SELECT 
"Product Name",
"Supplier Name",
"Consumer Name",
[Total Transactions Quantity]
FROM Tb_Transactions_Cube
WHERE "Supplier Name" IS NOT NULL
AND "Supplier City" ='Madison'
AND "Supplier State" IS NOT NULL
AND "Consumer Name" IS NOT NULL
AND "Consumer City" IS NOT NULL
AND "Consumer State" ='Illinois'
AND "Product Name" IS NOT NULL;

--9) Quantity of each product sold by supplier Bernstein to consumers in
--Chicago?
SELECT [Product Name],[Supplier Name],[Consumer City],
SUM([Total Transactions Quantity]) AS [Total Quantity]
FROM Tb_Transactions_Cube
WHERE 
[Supplier Name]='Bernstein'
AND[Supplier City] IS NOT NULL
AND[Supplier State] IS NOT NULL
AND[Consumer Name] IS NOT NULL
AND[Consumer City]='Chicago'
AND[Consumer State] IS NOT NULL
AND[Product Name] IS NOT NULL
GROUP BY[Product Name],[Supplier Name],[Supplier State],[Consumer City];


--10) Quantity of milk sold by supplier Bernstein to each of his milk
--consumers in Chicago?
SELECT 
"Consumer Name",
[Total Transactions Quantity]
FROM Tb_Transactions_Cube
WHERE "Supplier Name" ='Bernstein'
AND "Supplier City" IS NOT NULL
AND "Supplier State" IS NOT NULL
AND "Consumer Name" IS NOT NULL
AND "Consumer City" ='Chicago'
AND "Consumer State" IS NOT NULL
AND "Product Name" ='Milk';


--11) (Extra Credit 2%) For each product list quantity sold by suppliers in
--Madison to consumers in Chicago versus quantity sold by suppliers in
--Chicago to consumers in Madison (result columns will be: product name,
--quantity Madison_Chicago, quantity Chicago_Madison)?
SELECT 
MACH.PRODUCT_NAME,
isnull (MACH.[quantity Madison_Chicago],0) as "Quantity_Madison_Chicago",
isnull (CHMA.[quantity Chicago_Madison],0) as "Quantity_Chicago_Madison"
FROM
(SELECT
"Product Name" "PRODUCT_NAME",
[Total Transactions Quantity] "quantity Madison_Chicago"
FROM Tb_Transactions_Cube
WHERE "Supplier Name" IS NULL
AND "Supplier City" = 'Madison'
AND "Supplier State" IS NOT NULL
AND "Consumer Name" IS NULL
AND "Consumer City" ='Chicago'
AND "Consumer State" IS NOT NULL
AND "Product Name" IS NOT NULL) MACH
FULL OUTER JOIN
(SELECT
"Product Name" "PRODUCT_NAME",
[Total Transactions Quantity] "quantity Chicago_Madison"
FROM Tb_Transactions_Cube
WHERE "Supplier Name" IS NULL
AND "Supplier City" = 'Chicago'
AND "Supplier State" IS NOT NULL
AND "Consumer Name" IS NULL
AND "Consumer City" ='Madison'
AND "Consumer State" IS NOT NULL
AND "Product Name" IS NOT NULL) CHMA
ON
MACH.PRODUCT_NAME=CHMA.PRODUCT_NAME;
