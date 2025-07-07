local common = import 'common.jsonnet';

{
  name: 'fy3',
  filename: 'fy3.json',
  description: "The FY-3 data come from the MWHS microwave radiometer on China's Fengyun satellites. Data is available for three FY-3 satellites, FY-3A, FY-3B and FY-3C",
  title: 'FY-3',
  unique_id: '10',
  start_datetime: '2008-07-01T00:19:46',
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
    dataset_name: 'MICROWAVE_FCDR_V1.1-20200512/FY3/*/*.nc',
    type: 'application/vnd+netcdf',
    description: 'Observation dataset',
    locations: [common.hpc.hpc2020],
    size: '0.5 TB',
    inodes: '10',
    roles: ['data'],
  },
}
