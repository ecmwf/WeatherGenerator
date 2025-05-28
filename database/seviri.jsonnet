{
  name: "SEVIRI",
  filename: "seviri.json",
  description: "The Spinning Enhanced Visible and InfraRed Imager (SEVIRI) is an onboard sensor of the Meteosat Second Generation (MSG) satellites operated by EUMETSAT. SEVIRI provides high-frequency geostationary observations of the Earth’s atmosphere, land, and ocean surfaces over Europe, Africa, and parts of the Atlantic. ",
  title: "SEVIRI",
  unique_id: "4",
  start_datetime: "2018-02-12T21:45:00",
  end_datetime: "2023-03-21T07:45:00",
  frequency: "1h",
  fixed_timesteps: "True", 
  keywords: [
    "atmosphere",
    "observation",
	  "geostationary",
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

  // Retrieved with: for i, v  in enumerate(root.data.attrs["colnames"]): print(f"\"{v}\": 
  // [{root.data.attrs["mins"][i]}, {root.data.attrs["maxs"][i]}, {root.data.attrs["means"][i]}, 
  // {root.data.attrs["stds"][i]}],") 
  variables: {
    "healpix_idx_8": [0.0, 767.0, 344.7552795410156, 214.89877319335938],
    "lat": [-66.3325424194336, 66.4511489868164, 0.0, 90.0],
    "lon": [-67.47135925292969, 67.34668731689453, 0.0, 180.0],
    "zenith": [0.23000000417232513, 77.73999786376953, 0.0, 90.0],
    "solar_zenith": [0.20000000298023224, 179.8000030517578, 0.0, 180.0],
    "obsvalue_rawbt_4 (IR3.9)": [80.0, 335.70001220703125, 282.11981201171875, 16.33513641357422],
    "obsvalue_rawbt_5 (WV6.2)": [80.19999694824219, 263.29998779296875, 237.96363830566406, 8.569162368774414],
    "obsvalue_rawbt_6 (WV7.3)": [80.0, 287.70001220703125, 254.1988983154297, 11.519951820373535],
    "obsvalue_rawbt_7 (IR8.7)": [80.69999694824219, 330.79998779296875, 277.3443603515625, 17.72325897216797],
    "obsvalue_rawbt_8 (IR9.7)": [80.0999984741211, 301.29998779296875, 257.89312744140625, 13.6570463180542],
    "obsvalue_rawbt_9 (IR10.8)": [80.0, 335.6000061035156, 278.9452209472656, 18.87522315979004],
    "obsvalue_rawbt_10 (IR12.0)": [80.9000015258789, 335.6000061035156, 277.3193359375, 18.94614601135254],
    "obsvalue_rawbt_11 (IR13.4)": [80.19999694824219, 291.8999938964844, 257.7302551269531, 13.088528633117676],
    "cos_julian_day": [-1.0, 1.0, 0.0, 1.0],
    "sin_julian_day": [-1.0, 1.0, 0.0, 1.0],
    "cos_local_time": [-1.0, 1.0, 0.0, 1.0],
    "sin_local_time": [-1.0, 1.0, 0.0, 1.0],
    "cos_sza": [0.0, 1.0, 0.0, 1.0],
    "cos_latitude": [0.399530827999115, 0.9999936819076538, 0.0, 1.0],
    "sin_latitude": [-0.9158907532691956, 0.9167197346687317, 0.0, 1.0],
    "cos_longitude": [0.38314518332481384, 0.9999938011169434, 0.0, 1.0],
    "sin_longitude": [-0.9236881136894226, 0.9228522181510925, 0.0, 1.0],
    "cos_vza": [0.21234826743602753, 0.9999919533729553, 0.0, 1.0],
  },
  
  geometry: [-67.47135925292969, 67.34668731689453, -66.3325424194336, 66.4511489868164], 

  dataset: {
    dataset_name: "observations-od-ai-0001-2018-2023-meteosat-11-seviri-v1.zarr", 
    type: "application/vnd+zarr",
    description: "Observation dataset", 
    locations: ["HPC2020", "Leonardo"],
    size: "106GB",
    inodes: "2727",
    roles: ["data"]
  }
}