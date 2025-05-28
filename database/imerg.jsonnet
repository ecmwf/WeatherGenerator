{
  name: "IMERG",
  filename: "imerg.json",
  description: "NASA's Integrated Multi-satellitE Retrievals for GPM (IMERG) product combines information from the GPM satellite constellation to estimate precipitation over the majority of the Earth's surface. ",
  title: "IMERG", 
  unique_id: "5",
  start_datetime: "1998-01-01T06:00:00",
  end_datetime: "2024-07-31T18:00:00",
  frequency: "6h",
  fixed_timesteps: "True", 
  keywords: [
    "atmosphere",
    "precipitation",
	  "reanalysis",
    "global"
  ],
  providers: [
    {
      "name": "ECMWF",
          "roles": [ "host"], 
          "url": "https://ecmwf.int" 
    }
    {
          "name": "NASA",
          "roles": ["provider"],
          "url": "https://www.nasa.gov"
        }
  ],
 
  variables: {
    "tp":	[0,	0.814545,	0.00067628,	0.00326012, -6.54337427e-10, 0.00350661]
  },
  
  geometry: [-180, 180, -90,90], 

  dataset: {
    dataset_name: "nasa-imerg-grib-n320-1998-2024-6h-v1.zarr", 
    type: "application/vnd+zarr",
    description: "Anemoi dataset", 
    locations: ["HPC2020", "EWC", "JSC"],
    size: "18 GiB",
    inodes: "38,966",
    roles: ["data"]
  }
}