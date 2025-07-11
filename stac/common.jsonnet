// Variable filler

{
  providers: {
    copernicus: {
      name: 'Copernicus',
      roles: ['provider'],
      url: 'https://copernicus.eu',
    },
    ecmwf_host: {
      name: 'ECMWF',
      roles: ['host'],
      url: 'https://ecmwf.int',
    },
    ecmwf_provider: {
      name: 'ECMWF',
      roles: ['provider'],
      url: 'https://www.ecmwf.int/',
    },
    nasa: {
      name: 'NASA',
      roles: ['provider'],
      url: 'https://www.nasa.gov',
    },
    nasa_processor: {
      name: 'NASA',
      roles: ['processor'],
      url: 'https://www.nasa.gov',
    },    
    eumetsat: {
      name: 'EUMETSAT',
      roles: ['provider'],
      url: 'https://eumetsat.int',
    },
    eumetsat_processor: {
      name: 'EUMETSAT',
      roles: ['processor'],
      url: 'https://eumetsat.int',
    },
    cma: {
      name: 'CMA',
      roles: ['provider'],
      url: 'https://www.cma.gov.cn/',
    },
  },
  hpc: {
    leonardo: 'leonardo',
    hpc2020: 'hpc2020',
    lumi: 'lumi',
    ewc: 'European Weather Cloud',
    marenostrum5: 'marenostrum5',
    jsc: 'juwels_booster',
    levante: 'levante',
    alps: 'alps',
  },
}
