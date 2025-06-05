local common = import 'common.jsonnet';

{
  name: 'SEVIRI',
  filename: 'seviri.json',
  description: 'The Spinning Enhanced Visible and InfraRed Imager (SEVIRI) is an onboard sensor of the Meteosat Second Generation (MSG) satellites operated by EUMETSAT. SEVIRI provides high-frequency geostationary observations of the Earth’s atmosphere, land, and ocean surfaces over Europe, Africa, and parts of the Atlantic. ',
  title: 'SEVIRI',
  unique_id: '4',
  start_datetime: '2018-02-12T21:45:00',
  end_datetime: '2023-03-21T07:45:00',
  frequency: '1h',
  fixed_timesteps: 'True',
  keywords: [
    'atmosphere',
    'observation',
    'geostationary',
    'satellite',
  ],
  providers: [
    common.providers.ecmwf_host,
    common.providers.eumetsat,
  ],

  // Retrieved with: root.data.attrs["colnames"]),
  // {root.data.attrs["mins"],root.data.attrs["maxs"], root.data.attrs["means"],
  // {root.data.attrs["stds"]
  variables: {
    names: ['healpix_idx_8', 'lat', 'lon', 'zenith', 'solar_zenith', 'obsvalue_rawbt_4  (IR3.9)', 'obsvalue_rawbt_5  (WV6.2)', 'obsvalue_rawbt_6 (WV7.3)', 'obsvalue_rawbt_7 (IR8.7)', 'obsvalue_rawbt_8 (IR9.7)', 'obsvalue_rawbt_9 (IR10.8)', 'obsvalue_rawbt_10 (IR12.0)', 'obsvalue_rawbt_11 (IR13.4)', 'cos_julian_day', 'sin_julian_day', 'cos_local_time', 'sin_local_time', 'cos_sza', 'cos_latitude', 'sin_latitude', 'cos_longitude', 'sin_longitude', 'cos_vza'],
    mins: [0.0, -66.3325424194336, -67.47135925292969, 0.23000000417232513, 0.20000000298023224, 80.0, 80.19999694824219, 80.0, 80.69999694824219, 80.0999984741211, 80.0, 80.9000015258789, 80.19999694824219, -1.0, -1.0, -1.0, -1.0, 0.0, 0.399530827999115, -0.9158907532691956, 0.38314518332481384, -0.9236881136894226, 0.21234826743602753],
    maxs: [767.0, 66.4511489868164, 67.34668731689453, 77.73999786376953, 179.8000030517578, 335.70001220703125, 263.29998779296875, 287.70001220703125, 330.79998779296875, 301.29998779296875, 335.6000061035156, 335.6000061035156, 291.8999938964844, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9999936819076538, 0.9167197346687317, 0.9999938011169434, 0.9228522181510925, 0.9999919533729553],
    means: [344.7552795410156, 0.0, 0.0, 0.0, 0.0, 282.11981201171875, 237.96363830566406, 254.1988983154297, 277.3443603515625, 257.89312744140625, 278.9452209472656, 277.3193359375, 257.7302551269531, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    stds: [214.89877319335938, 90.0, 180.0, 90.0, 180.0, 16.33513641357422, 8.569162368774414, 11.519951820373535, 17.72325897216797, 13.6570463180542, 18.87522315979004, 18.94614601135254, 13.088528633117676, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  },

  geometry: [-67.47135925292969, 67.34668731689453, -66.3325424194336, 66.4511489868164],

  dataset: {
    dataset_name: 'observations-od-ai-0001-2018-2023-meteosat-11-seviri-v1.zarr',
    type: 'application/vnd+zarr',
    description: 'Observation dataset',
    locations: [common.hpc.hpc2020, common.hpc.leonardo],
    size: '106 GB',
    inodes: '2727',
    roles: ['data'],
  },
}
