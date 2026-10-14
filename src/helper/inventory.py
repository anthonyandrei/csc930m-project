import os                                                                                                                                                
import platform                                                                                                                                          
import sys                                                                                                                                               
import importlib.metadata                                                                                                                                
import pandas as pd                                                                                                                                      
from PIL import Image                                                                                                                                    
                                                                                                                                                            
EXPECTED_TOTAL_RECORDS = 9122                                                                                                                            
EXPECTED_TRAIN_RECORDS = 8346                                                                                                                            
EXPECTED_TEST_RECORDS = 776                                                                                                                              
                                                                                                                                                            
EXPECTED_CLASSES = {                                                                                                                                     
    "asphalt",                                                                                                                                           
    "concrete",                                                                                                                                          
    "paving stones",                                                                                                                                     
    "sett",                                                                                                                                              
    "unpaved",                                                                                                                                           
}                                                                                                                                                        
                                                                                                                                                            
EXPECTED_COLUMNS = [                                                                                                                                     
    "mapillary_image_id",                                                                                                                                
    "user_id",                                                                                                                                           
    "user_name",                                                                                                                                         
    "captured_at",                                                                                                                                       
    "longitude",                                                                                                                                         
    "latitude",                                                                                                                                          
    "train",                                                                                                                                             
    "surface_type",                                                                                                                                      
    "surface_quality",                                                                                                                                   
]                                                                                                                                                        
                                                                                                                                                            
                                                                                                                                                            
def get_environment_info():                                                                                                                              
    packages = ["numpy", "pandas", "opencv-python", "pillow"]                                                                                            
    versions = {}                                                                                                                                        
    for pkg in packages:                                                                                                                                 
        try:                                                                                                                                             
            versions[pkg] = importlib.metadata.version(pkg)                                                                                              
        except importlib.metadata.PackageNotFoundError:                                                                                                  
            versions[pkg] = "not installed"                                                                                                              
                                                                                                                                                            
    return {                                                                                                                                             
        "python_version": sys.version.split()[0],                                                                                                        
        "os": f"{platform.system()} {platform.release()}",                                                                                               
        "packages": versions,                                                                                                                            
    }                                                                                                                                                    
                                                                                                                                                            
                                                                                                                                                            
def load_and_validate_metadata(csv_path):                                                                                                                
    if not os.path.exists(csv_path):                                                                                                                     
        raise FileNotFoundError(f"Metadata file not found at: {csv_path}")                                                                               
                                                                                                                                                            
    df = pd.read_csv(csv_path, dtype={"mapillary_image_id": str})                                                                                        
                                                                                                                                                            
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)                                                                                               
    if missing_cols:                                                                                                                                     
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")                                                                             
                                                                                                                                                            
    total_rows = len(df)                                                                                                                                 
    train_count = int(df["train"].sum())                                                                                                                 
    test_count = total_rows - train_count                                                                                                                
    unique_ids = df["mapillary_image_id"].nunique()                                                                                                      
    null_counts = df.isnull().sum().to_dict()                                                                                                            
                                                                                                                                                            
    classes_found = set(df["surface_type"].dropna().unique())                                                                                            
    unexpected_classes = classes_found - EXPECTED_CLASSES                                                                                                
                                                                                                                                                            
    audit_summary = {                                                                                                                                    
        "total_records": total_rows,                                                                                                                     
        "expected_total": EXPECTED_TOTAL_RECORDS,                                                                                                        
        "train_records": train_count,                                                                                                                    
        "expected_train": EXPECTED_TRAIN_RECORDS,                                                                                                        
        "test_records": test_count,                                                                                                                      
        "expected_test": EXPECTED_TEST_RECORDS,                                                                                                          
        "unique_ids": unique_ids,                                                                                                                        
        "duplicate_ids": total_rows - unique_ids,                                                                                                        
        "has_nulls": any(v > 0 for v in null_counts.values()),                                                                                           
        "unexpected_classes": list(unexpected_classes),                                                                                                  
    }                                                                                                                                                    
                                                                                                                                                            
    return df, audit_summary                                                                                                                             
                                                                                                                                                            
                                                                                                                                                            
def audit_image_inventory(metadata_df, image_dir):                                                                                                       
    if not os.path.exists(image_dir):                                                                                                                    
        raise FileNotFoundError(f"Image directory not found at: {image_dir}")                                                                            
                                                                                                                                                            
    dir_files = set(os.listdir(image_dir))                                                                                                               
                                                                                                                                                            
    expected_ids = metadata_df["mapillary_image_id"].tolist()                                                                                            
    missing_ids = []                                                                                                                                     
    decode_failures = []                                                                                                                                 
    readable_paths = {}                                                                                                                                  
                                                                                                                                                            
    for img_id in expected_ids:                                                                                                                          
        filename = f"{img_id}.jpg"                                                                                                                       
        file_path = os.path.join(image_dir, filename)                                                                                                    
                                                                                                                                                            
        if filename not in dir_files:                                                                                                                    
            missing_ids.append(img_id)                                                                                                                   
            continue                                                                                                                                     
                                                                                                                                                            
        try:                                                                                                                                             
            with Image.open(file_path) as img:                                                                                                           
                img.verify()                                                                                                                             
            readable_paths[img_id] = file_path                                                                                                           
        except Exception as err:                                                                                                                         
            decode_failures.append((img_id, str(err)))                                                                                                   
                                                                                                                                                            
    extra_files = [                                                                                                                                      
        f                                                                                                                                                
        for f in dir_files                                                                                                                               
        if not f.startswith(".")                                                                                                                         
        and f not in {f"{i}.jpg" for i in expected_ids}                                                                                                  
    ]                                                                                                                                                    
                                                                                                                                                            
    inventory_df = metadata_df.copy()                                                                                                                    
    inventory_df["file_path"] = inventory_df["mapillary_image_id"].map(                                                                                  
        lambda x: os.path.join(image_dir, f"{x}.jpg")                                                                                                    
    )                                                                                                                                                    
    inventory_df["is_present"] = inventory_df["file_path"].map(os.path.exists)                                                                           
    inventory_df["is_readable"] = inventory_df["mapillary_image_id"].isin(                                                                               
        readable_paths                                                                                                                                   
    )                                                                                                                                                    
                                                                                                                                                            
    summary = {                                                                                                                                          
        "expected_images": len(expected_ids),                                                                                                            
        "found_images": len(expected_ids) - len(missing_ids),                                                                                            
        "readable_images": len(readable_paths),                                                                                                          
        "missing_images": len(missing_ids),                                                                                                              
        "decode_failures": len(decode_failures),                                                                                                         
        "extra_files": len(extra_files),                                                                                                                 
        "unusable_images": len(missing_ids) + len(decode_failures),                                                                                      
    }                                                                                                                                                    
                                                                                                                                                            
    return inventory_df, summary, missing_ids, decode_failures