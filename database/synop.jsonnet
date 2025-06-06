local common = import 'common.jsonnet';

{
  name: 'SYNOP',
  filename: 'synop.json',
  description: 'SYNOP (surface synoptic observation) data consist of standardized meteorological observations collected from land-based weather stations worldwide, typically at 6-hourly or hourly intervals. These observations include key atmospheric variables such as temperature, wind speed and direction, pressure, humidity, cloud cover, and precipitation. ',
  title: 'SYNOP',
  unique_id: '7',
  start_datetime: '1979-01-01T00:00:00',
  end_datetime: '2023-05-31T21:00:0',
  frequency: '3h',
  fixed_timesteps: 'False',
  keywords: [
    'atmosphere',
    'observation',
    'synoptic data',
  ],
  providers: [
    common.providers.ecmwf_host,
  ],

  // Retrieved with: root.data.attrs["colnames"],
  // root.data.attrs["mins"], root.data.attrs["maxs"],
  // root.data.attrs["means"], root.data.attrs["stds"]
  variables: {

    names: ['healpix_idx_8', 'seqno', 'lat', 'lon', 'stalt', 'lsm', 'obsvalue_tsts_0', 'obsvalue_t2m_0', 'obsvalue_u10m_0', 'obsvalue_v10m_0', 'obsvalue_rh2m_0', 'obsvalue_ps_0', 'cos_julian_day', 'sin_julian_day', 'cos_local_time', 'sin_local_time', 'cos_sza', 'cos_latitude', 'sin_latitude', 'cos_longitude', 'sin_longitude'],
    mins: [0.0, 5704.0, -90.0, -180.0, -389.0, 0.0, 229.5500030517578, 184.3000030517578, -55.149234771728516, -51.21000289916992, 1.1888814687225063e-14, 15990.0, -1.0, -0.9999994039535522, -1.0, -1.0, 0.0, -4.371138828673793e-08, -1.0, -1.0, -1.0],
    maxs: [767.0, 22173636.0, 90.0, 180.0, 31072.0, 1.0, 320.20001220703125, 338.0, 80.0, 51.645042419433594, 1.0, 113770.0, 1.0, 0.9999994039535522, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    means: [211.5513153076172, 5340427.5, 0.0, 0.0, 307.2554626464844, 0.7098972201347351, 291.65081787109375, 285.33074951171875, 0.2072220742702484, 0.05550207942724228, 0.7196683883666992, 97822.171875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    stds: [208.08619689941406, 4946647.5, 90.0, 180.0, 532.384521484375, 0.3707307279109955, 8.882925033569336, 13.912480354309082, 3.423595905303955, 3.289386034011841, 0.20914055407047272, 5907.458984375, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

  },

  geometry: [-180, 180, -90, 90],

  dataset: {
    dataset_name: 'observations-ea-ofb-0001-1979-2023-combined-surface-v2',
    type: 'application/vnd+zarr',
    description: 'Observation dataset',
    locations: [common.hpc.hpc2020, common.hpc.lumi],
    size: '61.5 GB',
    inodes: '4711',
    roles: ['data'],
  },
}
