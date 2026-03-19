-- ----------------------------------------
-- Properties Table
-- ----------------------------------------

CREATE TABLE properties (
    property_id SERIAL PRIMARY KEY,
    address TEXT,
    metro_area TEXT,
    sq_footage INTEGER,
    property_type TEXT
);

-- ----------------------------------------
-- Financials Table
-- ----------------------------------------

CREATE TABLE financials (
    property_id INTEGER,
    revenue NUMERIC,
    net_income NUMERIC,
    expenses NUMERIC,

    FOREIGN KEY (property_id)
    REFERENCES properties(property_id)
);



-- ----------------------------------------
-- Insert Sample Property Data
-- ----------------------------------------

INSERT INTO properties (address, metro_area, sq_footage, property_type)
VALUES
('123 Main St', 'Chicago', 12000, 'Industrial'),
('45 Lake Shore Dr', 'Chicago', 9000, 'Office'),
('78 Market St', 'San Francisco', 8500, 'Retail'),
('200 Sunset Blvd', 'Los Angeles', 15000, 'Industrial'),
('88 Ocean Ave', 'Miami', 7000, 'Retail');


-- ----------------------------------------
-- Insert Sample Financial Data
-- ----------------------------------------

INSERT INTO financials (property_id, revenue, net_income, expenses)
VALUES
(1,1200000,450000,750000),
(2,900000,300000,600000),
(3,850000,280000,570000),
(4,1500000,520000,980000),
(5,720000,210000,510000);