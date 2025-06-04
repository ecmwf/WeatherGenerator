local common = import 'common.jsonnet';

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
    common.providers.ecmwf_host, 
    common.providers.nasa, 
  ],
 
  variables: {
    names: ["tp"], 
    mins: [0],
    maxs:	[0.814545],	
    means: [0.00067628],
    stds: [0.00326012],
    tendencies:
    {
      means: [-6.54337427e-10],
      stds: [0.00350661]
    }
  },
  
  geometry: [-180, 180, -90,90], 

  dataset: {
    dataset_name: "nasa-imerg-grib-n320-1998-2024-6h-v1.zarr", 
    type: "application/vnd+zarr",
    description: "Anemoi dataset", 
    locations: [common.hpc.hpc2020, common.hpc.ewc, common.hpc.jsc],
    size: "18 GiB",
    inodes: "38,966",
    roles: ["data"]
  }
}