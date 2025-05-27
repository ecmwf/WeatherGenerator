// import functions
local fn = import 'functions.libsonnet';

// URL for hrefs
local href_link = "https://raw.githubusercontent.com/ecmwf/WeatherGenerator/refs/heads/iluise/develop/stac-database/database/folder";

local era5v8 = import "era5_v8.jsonnet";

// Dataset: ERA5
// local era5 = {
//   name: "ERA5v8",
//   filename: "era5v8.json",
//   description: "ERA5 is a reanalysis dataset produced by ECMWF, providing hourly estimates of a large number of atmospheric, land, and oceanic climate variables.",
//   unique_id: 0,
//   title: "ERA5 Dataset",
//   start_datetime: "1950-01-01T00:00:00Z",
//   end_datetime: "2023-12-31T23:59:59Z",
//   keywords: [
//     "polar orbiter",
//     "satellite",
//     "atmosphere",
//     "observations"
//   ],
//   providers: [
//     {
//       name: "ECMWF",
//       roles: ["producer", "licensor"],
//       url: "https://www.ecmwf.int/"
//     }
//   ],
//   variables: {
//     "2t": ["2m temperature in Kelvin", 0, 1, 5, 10, 0],
//     "10u": ["10m u component of wind in m/s", 0, 1, 5, 10, 0],
//     "10v": ["10m v component of wind in m/s", 0, 2, 3, 4, 0],
//     "tp": ["Total precipitation in kg/m^2", 5, 6, 7, 8, 0],
//     "sp": ["Surface pressure in Pa", 0, 5, 6, 7, 0]
//   },
  
//   geometry: [-180, 180, -90, 90], 
  
//   ref_links: [
//     {
//     rel: "DOC",
//     href: "https://confluence.ecmwf.int/display/MAEL/WeatherGenerator+Dataset",
//     title: "ERA5 Documentation (ECMWF Confluence)"
//     }
//   ],

//   dataset: {
//     dataset_name: "aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr", 
//     type: "application/vnd+zarr",
//     notes: "ERA5 data on O96 healPix grid. version 8. Contains tendencies", 
//     version: "v8",
//     locations: "HPC2020, MareNostrum5, Leonardo",
//     size: "1.2 TB",
//     inodes: "1000000",
//     roles: "data"
//   }
// };

local datasets = [era5v8];

local files = [ds.filename + ".json" for ds in datasets];
{ "catalogue.json":
    {
        "type": "Catalog",
        "id": "weathergen",
        "stac_version": "1.0.0",
        "description": "The data catalogue of the WeatherGenerator project",
        
        "links": [fn.dataset_entry_catalogue(ds, href_link) for ds in datasets],
        
        "stac_extensions": [
        "https://stac-extensions.github.io/datacube/v2.2.0/schema.json",
        "https://stac-extensions.github.io/alternate-assets/v1.2.0/schema.json"
        ],
        "title": "The WeatherGenerator data catalogue",
    }
}

{
    [ds.filename + ".json"]: fn.dataset_entry_fill(ds)
    for ds in datasets
}
