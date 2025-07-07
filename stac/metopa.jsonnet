local common = import 'common.jsonnet';

{
  name: 'MHS METOP-A',
  filename: 'metopa.json',
  description: 'The MHS METOP-A dataset is derived from the Microwave Humidity Sounder instrument onboard the Meteorological Operational A satellite.',
  title: 'MHS METOP-A',
  unique_id: '9',
  start_datetime: '2006-10-31T21:24:14',
  end_datetime: '2018-12-31T23:46:05',
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
    dataset_name: 'MICROWAVE_FCDR_V1.1-20200512/METOPA/*/*.nc',
    type: 'application/vnd+netcdf',
    description: 'Observation dataset',
    locations: [common.hpc.hpc2020],
    size: '1.3 TB',
    inodes: '64637',
    roles: ['data'],
  },
}
