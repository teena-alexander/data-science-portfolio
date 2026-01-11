--a) (25 points) Based on the tables in the database given by the description
--use SQL with GROUP BY, CUBE and ROLLUP to create a cube.

SELECT DISTINCT S.Name "Supplier Name",
S.City "Supplier City",
S.State "Supplier State",
P.Name "Product Name",
P.Product_Packaging "Product Packaging",
P.Product_Category "Product Category",
P.Product_Line "Product Line",
SUM(Quantity) "Total Transactions Quantity",
SUM(Quantity*Price) "Total Transaction Value",
MAX(Price) "Max. Price",
MIN(Price) "Min. Price"
INTO Tb_SCO_Cube
FROM Tb_Supplier S,  Tb_Product P, Tb_Offers T
WHERE S.Supp_ID=T.Supp_ID AND
P.Prod_ID=T.Prod_ID
GROUP BY CUBE((S.State, S.City, S.Name),
(P.Product_Packaging, P.Name),(P.Product_Category,P.Product_Line,P.Name)),
ROLLUP(S.State, S.City, S.Name),
ROLLUP(P.Product_Packaging, P.Name)




--b) (25 points) Given the cube created at point a) solve the following queries
--using SQL:
--1. Value of products offered by supplier and by product packaging? (2 points)
SELECT
cub.[Supplier Name],
cub.[Product Packaging],
cub.[Total Transaction Value]
FROM
[dbo].[Tb_SCO_Cube] cub
WHERE
"Supplier Name" IS NOT NULL
AND "Supplier City" IS NOT NULL
AND "Supplier State" IS NOT NULL
AND "Product Name" IS NULL
AND "Product Packaging" IS NOT NULL
AND "Product Category" IS NULL
AND "Product Line" IS NULL


--2. Volume of milk offered by each supplier in Wisconsin? (2 points)
SELECT
cub.[Supplier Name],
cub.[Total Transactions Quantity]
FROM
[dbo].[Tb_SCO_Cube] cub
WHERE
"Supplier Name" IS NOT NULL
AND "Supplier City" IS NOT NULL
AND "Supplier State" ='Wisconsin'
AND "Product Name" ='Milk'
AND "Product Packaging" IS NULL
AND "Product Category" IS NOT NULL
AND "Product Line" IS NOT NULL


--3. Find the maximum price for each product offered in Madison? (5 points)
SELECT
cub.[Product Name],
MAX(cub.[Max. Price]) "MAXIMUM PRICE"

FROM
[dbo].[Tb_SCO_Cube] cub
WHERE
"Supplier Name" IS NOT NULL
AND "Supplier City" ='Madison'
AND "Supplier State" IS NOT NULL
AND "Product Name" IS NOT NULL
AND "Product Packaging" IS NULL
AND "Product Category" IS NOT NULL
AND "Product Line" IS NOT NULL
GROUP BY cub.[Product Name]

--4. For each supplier city find the product offered in largest quantity?(8 points)

SELECT
cub.[Supplier City],
cub.[Product Name],
cub.[Total Transactions Quantity]
FROM
[dbo].[Tb_SCO_Cube] cub
WHERE
"Supplier Name" IS NULL
AND "Supplier City" IS NOT NULL
AND "Supplier State" IS NOT NULL
AND "Product Name" IS NOT NULL
AND "Product Packaging" IS NULL
AND "Product Category" IS NOT NULL
AND "Product Line" IS NOT NULL
AND cub.[Total Transactions Quantity]>=  (SELECT 
										MAX([Total Transactions Quantity] ) AS "MAX QTY"
										FROM
										[dbo].[Tb_SCO_Cube]
										WHERE
										"Supplier Name" IS NULL
										AND "Supplier City" IS NOT NULL
										AND "Supplier State" IS NOT NULL
										AND "Product Name" IS NOT NULL
										AND "Product Packaging" IS NULL
										AND "Product Category" IS NOT NULL
										AND "Product Line" IS NOT NULL								
										AND "Supplier City"=cub.[Supplier City]
										GROUP BY "Supplier City")



--5. For each product find the city where it is offered at the lowest price?(8 points)
SELECT
cub.[Product Name],
cub.[Supplier City],
cub.[Min. Price]
FROM
[dbo].[Tb_SCO_Cube] cub
WHERE
"Supplier Name" IS NULL
AND "Supplier City" IS NOT NULL
AND "Supplier State" IS NOT NULL
AND "Product Name" IS NOT NULL
AND "Product Packaging" IS NULL
AND "Product Category" IS NOT NULL
AND "Product Line" IS NOT NULL

AND cub.[Min. Price] <=  (SELECT 
							Min([Min. Price] ) AS "MinPrice"
							FROM
							[dbo].[Tb_SCO_Cube]
							WHERE
							"Supplier Name" IS NULL
							AND "Supplier City" IS NOT NULL
							AND "Supplier State" IS NOT NULL
							AND "Product Name" IS NOT NULL
							AND "Product Packaging" IS NULL
							AND "Product Category" IS NOT NULL
							AND "Product Line" IS NOT NULL								
							AND cub.[Product Name]="Product Name"
							GROUP BY "Product Name")


