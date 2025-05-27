// import functions
local fn = import 'functions.libsonnet';

// URL for hrefs
local href_link = "https://raw.githubusercontent.com/ecmwf/WeatherGenerator/refs/heads/iluise/develop/stac-database/database/";

local era5v8 = import "era5_v8.jsonnet";

local datasets = [era5v8];

local files = [ds.filename + ".json" for ds in datasets];
{ "catalogue.json":
    {
        "type": "Catalog",
        "id": "weathergen",
        "stac_version": "1.0.0",
        "description": "The data catalogue of the WeatherGenerator project",
        
        "links": 
        [
            {
                "rel": "root",
                "href": href_link+"catalogue.json",
                "type": "application/json",
                "title": "The WeatherGenerator data server"
            },
            {
                "rel": "self",
                "href": href_link+"catalogue.json",
                "type": "application/json"
         },
        ]
        +
        [fn.dataset_entry_catalogue(ds, href_link) for ds in datasets],
        
        "stac_extensions": [
        "https://stac-extensions.github.io/datacube/v2.2.0/schema.json",
        "https://stac-extensions.github.io/alternate-assets/v1.2.0/schema.json"
        ],
        "title": "The WeatherGenerator data catalogue",
    }
}

{
    [ds.filename]: fn.dataset_entry_fill(ds)
    for ds in datasets
}
