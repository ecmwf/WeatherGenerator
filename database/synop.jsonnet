{
  name: "SYNOP",
  filename: "synop.json",
  description: "SYNOP (surface synoptic observation) data consist of standardized meteorological observations collected from land-based weather stations worldwide, typically at 6-hourly or hourly intervals. These observations include key atmospheric variables such as temperature, wind speed and direction, pressure, humidity, cloud cover, and precipitation. ",
  title: "SYNOP",
  unique_id: "7",
  start_datetime: "1979-01-01T00:00:00",
  end_datetime: "2023-05-31T21:00:0",
  frequency: "3h",
  fixed_timesteps: "False", 
  keywords: [
    "atmosphere",
    "observation",
	  "synoptic data",
  ],
  providers: [
    {
      "name": "ECMWF",
          "roles": [ "host"], 
          "url": "https://ecmwf.int" 
    }
  ],

  // Retrieved with: for i, v  in enumerate(root.data.attrs["colnames"]): print(f"\"{v}\": 
  // [{root.data.attrs["mins"][i]}, {root.data.attrs["maxs"][i]}, {root.data.attrs["means"][i]}, 
  // {root.data.attrs["stds"][i]}],") 
  variables: {
    "healpix_idx_8": [0.0, 767.0, 211.5513153076172,208.08619689941406],
    "seqno": [5704.0, 22173636.0, 5340427.5,4946647.5],
    "lat": [-90.0, 90.0, 0.0,90.0],
    "lon": [-180.0, 180.0, 0.0,180.0],
    "stalt": [-389.0, 31072.0, 307.2554626464844,532.384521484375],
    "lsm": [0.0, 1.0, 0.7098972201347351,0.3707307279109955],
    "obsvalue_tsts_0": [229.5500030517578, 320.20001220703125, 291.65081787109375,8.882925033569336],
    "obsvalue_t2m_0": [184.3000030517578, 338.0, 285.33074951171875,13.912480354309082],
    "obsvalue_u10m_0": [-55.149234771728516, 80.0, 0.2072220742702484,3.423595905303955],
    "obsvalue_v10m_0": [-51.21000289916992, 51.645042419433594, 0.05550207942724228,3.289386034011841],
    "obsvalue_rh2m_0": [1.1888814687225063e-14, 1.0, 0.7196683883666992,0.20914055407047272],
    "obsvalue_ps_0": [15990.0, 113770.0, 97822.171875,5907.458984375],
    "cos_julian_day": [-1.0, 1.0, 0.0,1.0],
    "sin_julian_day": [-0.9999994039535522, 0.9999994039535522, 0.0,1.0],
    "cos_local_time": [-1.0, 1.0, 0.0,1.0],
    "sin_local_time": [-1.0, 1.0, 0.0,1.0],
    "cos_sza": [0.0, 1.0, 0.0,1.0],
    "cos_latitude": [-4.371138828673793e-08, 1.0, 0.0,1.0],
    "sin_latitude": [-1.0, 1.0, 0.0,1.0],
    "cos_longitude": [-1.0, 1.0, 0.0,1.0],
    "sin_longitude": [-1.0, 1.0, 0.0,1.0],
  },
  
  geometry: [-180, 180, -90, 90], 

  dataset: {
    dataset_name: "observations-ea-ofb-0001-1979-2023-combined-surface-v2", 
    type: "application/vnd+zarr",
    description: "Observation dataset", 
    locations: ["HPC2020", "Lumi"],
    size: "61.5 GB",
    inodes: "4711",
    roles: ["data"]
  }
}