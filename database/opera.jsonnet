{
  name: "OPERA",
  filename: "opera.json",
  description: "The OPERA radar dataset is produced by the EUMETNET OPERA program, which coordinates and harmonizes European weather radar observations. It provides quality-controlled, pan-European radar composites and individual radar data from national meteorological services. ",
  title: "OPERA",
  unique_id: 3,
  start_datetime: "2013-01-22T15:05:00",
  end_datetime: "2024-02-15T14:05:00",
  frequency: "15m",
  keywords: [
    "radar",
    "precipitation",
	  "atmosphere",
	  "observations"
  ],
  providers: [
    {
      "name": "ECMWF",
      "roles": ["host"],
      "url": "https://ecmwf.int"
    }
  ],
  variables: {
    "mask":	[0,	3,	1.24214,	0.646755],
    "quality":	[0,	24.6,	0.233054,	0.195426],
    "tp":	[0,	1.09959e+19,	2.9961e+12,	2.78072e+15]
  },
  
  geometry: [-40, 58, 32, 68], 
  
  ref_links: [
    {
        "rel": "DOC",
        "href": "https://www.eumetnet.eu/wp-content/uploads/2017/01/OPERA_hdf_description_2014.pdf",
        "title": "EUMETNET OPERA program documentation"
      },
      {
        "rel": "collection",
        "href": "https://raw.githubusercontent.com/ecmwf/WeatherGenerator/refs/heads/iluise/develop/stac-database/database/catalogue.json",
        "type": "application/json"
      }
  ],

  dataset: {
    dataset_name: "rodeo-opera-files-2km-2013-2023-15m-v1-lambert-azimuthal-equal-area.zarr", 
    type: "application/vnd+zarr",
    description: "Anemoi dataset", 
    locations: ["HPC2020", "EWC", "MareNostrum5"],
    size: "959 GiB",
    inodes: "380,987",
    roles: ["data"]
  }
}