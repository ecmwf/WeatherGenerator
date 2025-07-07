local common = import 'common.jsonnet';

{
  name: 'metopa',
  filename: 'metopa.json',
  description: 'The METOP-A dataset is derived from the MHS instrument onboard the Meteorological Operational A satellite. MetOp-A, launched on 19 October 2006',
  title: 'METOP-A',
  unique_id: '9',
  start_datetime: '2007-01-01T00:36:13',
  end_datetime: '2018-12-31T23:58:08',
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
      'instrtemp',
      'scnlin',
      'satellite_azimuth_angle',
      'satellite_zenith_angle',
      'solar_azimuth_angle',
      'solar_zenith_angle',
      'data_quality_bitmask',
      'quality_scanline_bitmask',
      'time',
      'warmnedt',
      'coldnedt',
      'btemps',
      'u_independent_btemps',
      'u_structured_btemps',
      'u_common_btemps',
      'quality_issue_pixel_bitmask',
    ],

  },

  geometry: [-180, 180, -90, 90],

  dataset: {
    dataset_name: 'MICROWAVE_FCDR_V1.1-20200512/METOPA/*/*.nc',
    type: 'application/vnd+netcdf',
    description: 'Observation dataset',
    locations: [common.hpc.hpc2020, common.hpc.jsc],
    size: '0.5 TB',
    inodes: '10',
    roles: ['data'],
  },
}
