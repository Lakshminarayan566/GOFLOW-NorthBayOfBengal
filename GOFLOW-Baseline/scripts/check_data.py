import os
import xarray as xr

print("=== 1. CHECKING HYCOM NETCDF4 ===")
hycom_path = "data/hycom/dec2022_hycom.nc4"
if os.path.exists(hycom_path):
    try:
        ds_hycom = xr.open_dataset(hycom_path)
        print("✅ HYCOM loaded successfully!")
        print(f"   Variables present: {list(ds_hycom.data_vars)}")
        times = ds_hycom['time'].values
        print(f"   Time range: {times[0]} to {times[-1]}")
        print(f"   Total time steps: {len(times)}")
    except Exception as e:
        print(f"❌ Error reading HYCOM: {e}")
else:
    print("❌ HYCOM file not found at path.")

print("\n=== 2. CHECKING ERA5 GRIB ===")
era5_path = "data/era5/dec2022_era5.grib"
if os.path.exists(era5_path):
    try:
        import pygrib
        grbs = pygrib.open(era5_path)
        print("✅ ERA5 GRIB read successfully via pygrib!")
        
        # Pull a few sample messages to inspect variables and time structures
        messages = grbs.select()
        print(f"   Total data messages inside GRIB: {len(messages)}")
        
        # Print metadata from the first message to confirm variables
        sample_msg = messages[0]
        print(f"   Sample Variable Found: {sample_msg.name} ({sample_msg.shortName})")
        print(f"   Data Date/Time: {sample_msg.validDate}")
        grbs.close()
    except ImportError:
        print("❌ 'pygrib' is not installed. Run 'pip install pygrib' first.")
    except Exception as e:
        print(f"❌ Alternate GRIB read failed: {e}")
else:
    print("❌ ERA5 file not found at path.")

print("\n=== 3. CHECKING HIMAWARI DIRECTORY ===")
himaw_path = "data/himawari/dec2022_himawari"
if os.path.exists(himaw_path):
    files = os.listdir(himaw_path)
    print(f"✅ Himawari directory found.")
    print(f"   Total files in directory: {len(files)}")
    if len(files) > 0:
        print(f"   Sample file names: {files[:3]}")
else:
    print("❌ Himawari path not found.")