{
  name: "NPP-ATMS",
  filename: "npp-atms.json",
  description: "The NPP-ATMS (Advanced Technology Microwave Sounder) dataset is derived from the ATMS instrument onboard the NOAA/NASA National Polar-orbiting Partnership (NPP) satellite. It provides global measurements of atmospheric temperature, moisture, and pressure profiles, crucial for weather forecasting and climate monitoring",
  title: "NPP-ATMS",
  unique_id: "6",
  start_datetime: "2011-12-11T00:36:13",
  end_datetime: "2018-12-31T23:58:08",
  frequency: "NA",
  fixed_timesteps: "False", 
  keywords: [
    "atmosphere",
    "observation",
	  "polar-orbiter",
    "satellite"
  ],
  providers: [
    {
      "name": "ECMWF",
          "roles": [ "host"], 
          "url": "https://ecmwf.int" 
    }
    {
          "name": "EUMETSAT",
          "roles": ["provider"],
          "url": "https://eumetsat.int"
        }
  ],

  variables: {
    "quality_pixel_bitmask": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"], 
    "instrtemp": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],  
    "scnlin": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],       
    "satellite_azimuth_angle": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"], 
    "satellite_zenith_angle": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],  
    "solar_azimuth_angle": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],  
    "solar_zenith_angle": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],   
    "data_quality_bitmask": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],  
    "quality_scanline_bitmask": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],  
    "time": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],            
    "warmnedt": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],            
    "coldnedt": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],            
    "btemps": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],               
    "u_independent_btemps": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],    
    "u_structured_btemps": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],       
    "u_common_btemps": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],             
    "quality_issue_pixel_bitmask": ["TBA", "TBA", "TBA", "TBA", "TBA", "TBA"],   
  },
  
  geometry: [-180, 180, -90, 90], 
  
  ref_links: [
    {
        "rel": "DOC",
        "href": "https://user.eumetsat.int/catalogue/EO:EUM:DAT:0345",
        "title": "EUMETSAT documentation"
      },
      {
        "rel": "collection",
        "href": "https://raw.githubusercontent.com/ecmwf/WeatherGenerator/refs/heads/iluise/develop/stac-database/database/catalogue.json",
        "type": "application/json"
      }
  ],

  dataset: {
    dataset_name: "MICROWAVE_FCDR_V1.1-20200512/SNPP/*/*.nc", 
    type: "application/vnd+netcdf",
    description: "Observation dataset", 
    locations: ["HPC2020", "JSC"],
    size: "120GB",
    inodes: "",
    roles: ["data"]
  }
}