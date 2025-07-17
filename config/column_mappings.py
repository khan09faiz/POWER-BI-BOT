"""
Column mappings and standardization for equipment data
"""

# Standard column mappings - maps original column names to standardized names
COLUMN_MAPPINGS = {
    # Core identification
    "Cl.": "class_code",
    "Equipment": "equipment_id",
    "Equipment description": "equipment_description", 
    "Description of technical object": "technical_description",
    
    # Dates
    "Created on": "created_on",
    "Chngd On": "changed_on",
    "Date": "acquisition_date",
    "Date.1": "additional_date",
    "WtyEnd": "warranty_end",
    "GuarantDat": "guarantee_date",
    "DelDate": "delivery_date",
    "Start from": "start_date",
    "End-of-Use": "end_of_use_date",
    "Date of Last Goods Movemnt": "last_goods_movement",
    
    # Personnel
    "Created By": "created_by",
    "Changed By": "changed_by",
    
    # Financial
    "Acquisition Value": "acquisition_value",
    "Crcy": "currency",
    "ReplVal.": "replacement_value",
    "Provision fee": "provision_fee",
    "(Un)loadCosts": "unload_costs",
    
    # Manufacturer details
    "Manufacturer of Asset": "manufacturer",
    "Manufacturer drawing number": "manufacturer_drawing_number",
    "Manufacturer's Serial Number": "manufacturer_serial_number",
    "Model number": "model_number",
    
    # Physical properties
    "Size/dimension": "size_dimension",
    "Weight": "weight",
    "Un.": "unit",
    
    # Location and organization
    "Plnt": "plant",
    "SLoc": "storage_location",
    "Field": "field",
    "Object number": "object_number",
    "Object no.": "object_number_alt",
    
    # Technical specifications
    "Serial Number": "serial_number",
    "Serial Number.1": "serial_number_alt",
    "Inventory Number": "inventory_number",
    "Material": "material",
    "Config. material": "config_material",
    
    # Maintenance
    "MntPlan": "maintenance_plan",
    "MeasPoint": "measurement_point",
    "Master warranty": "master_warranty",
    "Warranty": "warranty",
    
    # Classification
    "Class": "equipment_class",
    "PGrp": "planning_group",
    "AGrp": "authorization_group",
    "ObjectType": "object_type",
    
    # Vendor information
    "Vendor": "vendor",
    "CurCustomer": "current_customer",
    
    # Status and control
    "RevLev": "revision_level",
    "ECN": "engineering_change_notice",
    "C": "control_indicator",
    "S": "status_indicator",
    "P": "processing_indicator",
    "R": "revision_indicator",
    "L": "location_indicator",
    "O": "operational_indicator",
}

# Data type mappings for optimization
DTYPE_MAPPINGS = {
    # Integer columns
    "equipment_id": "int64",
    "class_code": "int16",
    
    # Float columns  
    "acquisition_value": "float32",
    "replacement_value": "float32",
    "weight": "float32",
    "provision_fee": "float32",
    "unload_costs": "float32",
    
    # Category columns (for memory optimization)
    "manufacturer": "category",
    "plant": "category", 
    "field": "category",
    "equipment_class": "category",
    "object_type": "category",
    "currency": "category",
    "unit": "category",
    "vendor": "category",
    
    # String columns
    "equipment_description": "string",
    "technical_description": "string",
    "model_number": "string",
    "serial_number": "string",
    "inventory_number": "string",
}

# Columns to keep for core analysis (to reduce dimensionality)
CORE_COLUMNS = [
    "equipment_id",
    "equipment_description", 
    "created_on",
    "manufacturer",
    "acquisition_value",
    "weight",
    "plant",
    "field",
    "equipment_class",
    "maintenance_plan",
    "warranty_end",
    "serial_number",
    "model_number",
    "object_type"
]

# Columns that are likely to have high cardinality (for special handling)
HIGH_CARDINALITY_COLUMNS = [
    "equipment_id",
    "serial_number", 
    "manufacturer_serial_number",
    "inventory_number",
    "object_number",
    "manufacturer_drawing_number"
]

# Date columns for parsing
DATE_COLUMNS = [
    "created_on",
    "changed_on", 
    "acquisition_date",
    "warranty_end",
    "guarantee_date",
    "delivery_date",
    "start_date",
    "end_of_use_date",
    "last_goods_movement"
]
