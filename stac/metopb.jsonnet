local common = import 'common.jsonnet';

{
  name: 'Metop-B, MHS',
  filename: 'metopb.json',
  description: 'The MHS Metop-B dataset is derived from the Microwave Humidity Sounder instrument onboard the Meteorological Operational B satellite.',
  title: 'Metop-B, MHS',
  unique_id: '9',
  start_datetime: '2013-04-01T02:06:10',
  end_datetime: '2018-12-31T23:11:48',
  frequency: 'NA',
  fixed_timesteps: 'False',
  keywords: [
    'atmosphere',
    'observation',
    'polar-orbiter',
    'satellite',
  ],
  providers: [
    common.providers.ecmwf_host,
    common.providers.eumetsat,
  ],
  processing_level: '1C',
  provenance: 'EUMETSAT',


  variables: {
    names: [
      'quality_pixel_bitmask',
      'btemps',
      'instrtemp',
      'scnlin',
      'satellite_azimuth_angle',
      'satellite_zenith_angle',
      'solar_azimuth_angle',
      'solar_zenith_angle',
      'u_independent_btemps',
      'u_structured_btemps',
      'quality_issue_pixel_bitmask',
      'data_quality_bitmask',
      'quality_scanline_bitmask',
      'u_common_btemps',
      'warmnedt',
      'coldnedt',
      'time',
    ],

  },

  geometry: [-180, 180, -90, 90],

  dataset: {
    dataset_name: 'MICROWAVE_FCDR_V1.1-20200512/METOPB/*/*.nc',
    type: 'application/vnd+netcdf',
    description: 'Observation dataset',
    locations: [common.hpc.hpc2020],
    size: '634.1 GB',
    inodes: '31708',
    roles: ['data'],
  },
}
