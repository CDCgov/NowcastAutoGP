# NowcastAutoGP.jl

*Centers for Disease Control and Prevention • Center for Forecasting and Outbreak Analytics*

**Automated Gaussian Process model discovery for time series data with revisions**


## About NowcastAutoGP.jl

**NowcastAutoGP.jl** is a specialized Julia package developed by CDC's Center for Forecasting and Outbreak Analytics for automated nowcasting of epidemiological time series data using Gaussian processes. This tool directly supports CFA's core functions of:

- **Real-time epidemic monitoring** and trend analysis
- **Outbreak response analytics** with uncertainty quantification
- **Public health forecasting** for resource planning and policy decisions
- **Data-driven decision support** during health emergencies

Nowcasting in public health refers to estimating current disease activity while accounting for reporting delays common in surveillance systems. This capability is essential for CFA's mission to provide timely, actionable intelligence during public health responses.


## Installation

```julia
# Standard installation
using Pkg
Pkg.add(url="https://github.com/CDCgov/NowcastAutoGP.jl")
```

## API Reference and Resources

- **[Complete API Documentation](api.md)**

---

## Legal and Open Government

This repository constitutes a **work of the United States Government** and is not subject to domestic copyright protection under 17 USC § 105. This repository is in the **public domain** within the United States, supporting CFA's commitment to open science and collaborative public health analytics.

**CFA Disclaimer**: This repository was created by CDC's Center for Forecasting and Outbreak Analytics to support advanced analytics in public health surveillance and response. The tools and methodologies represent CFA's commitment to evidence-based, collaborative approaches to protecting public health.

---

*Developed by CDC's Center for Forecasting and Outbreak Analytics*
*Advanced Analytics • Effective Decisions • Public Health Response*
*Learn more: [cdc.gov/forecast-outbreak-analytics](https://www.cdc.gov/forecast-outbreak-analytics/)*

## Key Features for Public Health Applications

### 🔬 **Epidemiologically-Informed Modeling**
- **Gaussian Process regression** optimized for surveillance data patterns
- **Built-in reporting delay models** for common surveillance systems
- **Uncertainty quantification** critical for risk assessment and policy decisions
- **Bayesian framework** enabling incorporation of prior epidemiological knowledge

### 📊 **CDC-Compatible Data Integration**
- **Automated data transformation** including Box-Cox, logit, and epidemiological transforms
- **Flexible data ingestion** supporting multiple surveillance system formats
- **Type-safe design** preventing common errors in epidemiological analysis
- **Nowcast integration** with proper uncertainty propagation

### ⚡ **Operational Excellence**
- **Automated model selection** minimizing manual intervention
- **Performance optimized** for real-time surveillance environments
- **Scalable implementation** handling large surveillance datasets
- **Parallel processing** support for ensemble forecasting

### 🔒 **Government-Grade Standards**
- **Comprehensive testing** and validation for operational reliability
- **Reproducible methodology** meeting scientific and regulatory standards
- **Security-conscious design** appropriate for government computing environments
- **Open source transparency** enabling peer review and collaboration

## Installation and Setup

For CDC and government installations, ensure compliance with institutional security requirements:

```julia
# Standard installation from Julia General Registry
using Pkg
Pkg.add("NowcastAutoGP")

# For development or testing environments
Pkg.add(url="https://github.com/CDCgov/NowcastAutoGP.jl")
```

### System Requirements

- **Julia 1.8+** (LTS recommended for production surveillance systems)
- **Government computing environment** compatibility
- **Minimal dependencies** for security and maintenance

## Public Health Quick Start

Here's an example for nowcasting influenza-like illness (ILI) surveillance data:

```julia
using NowcastAutoGP
using Dates

# Example: Weekly ILI surveillance data with reporting delays
surveillance_dates = Date(2023, 10, 1):Week(1):Date(2024, 3, 1)
ili_rates = [2.1, 2.3, 2.8, 3.1, 3.7, 4.2, 4.8, 5.1, 4.9, 4.5, 4.1, 3.8,
             3.4, 3.0, 2.7, 2.4, 2.2, 2.0, 1.9, 1.8, missing, missing]

# Create epidemiologically-appropriate transformed data
# Log transformation common for disease rates
tdata = create_transformed_data(surveillance_dates, ili_rates;
                               transformation = log,
                               handle_missing = true)

# Fit Gaussian Process model optimized for surveillance data
gp_model = make_and_fit_model(tdata)

# Generate nowcast for current time period (accounting for reporting delay)
current_date = Date(2024, 3, 1)
nowcast_horizon = Week(2)  # Account for 2-week reporting delay

nowcast_results = forecast(gp_model,
                          [current_date - nowcast_horizon, current_date],
                          1000;  # Monte Carlo samples for uncertainty
                          inv_transformation = exp)

# Extract epidemiologically-relevant outputs
current_ili_estimate = nowcast_results.mean[end]
credible_interval_95 = (nowcast_results.quantile_05[end],
                       nowcast_results.quantile_95[end])

println("Current ILI activity estimate: $(round(current_ili_estimate, digits=2))%")
println("95% credible interval: $(round.(credible_interval_95, digits=2))")
```

## Epidemiological Use Cases

### **Infectious Disease Surveillance**
- **Influenza monitoring**: Weekly ILI and laboratory-confirmed cases
- **COVID-19 surveillance**: Hospitalizations, deaths, and wastewater signals
- **Foodborne illness**: Outbreak detection with reporting delays
- **Vector-borne diseases**: Seasonal activity monitoring

### **Emergency Response Applications**
- **Outbreak detection**: Early warning systems for anomalous activity
- **Resource planning**: Hospital capacity and supply chain management
- **Intervention assessment**: Real-time evaluation of public health measures
- **Situational awareness**: Current disease burden during emergencies

### **Surveillance System Enhancement**
- **Retrospective analysis**: Correcting for historical reporting delays
- **Data quality assessment**: Identifying systematic reporting issues
- **Multi-source integration**: Combining traditional and novel data streams
- **Forecasting validation**: Evaluating prediction system performance

## Mathematical Framework for Epidemiologists

The package implements Gaussian Process regression specifically adapted for epidemiological time series:

**Model Structure:**
```
y(t) = f(t) + ε(t)
```

Where:
- `y(t)`: Observed surveillance data (possibly transformed)
- `f(t)`: Latent disease activity (Gaussian process)
- `ε(t)`: Observation noise (reporting errors, measurement uncertainty)

**Key Adaptations for Public Health:**
- **Epidemiological priors**: Informed kernel selection for disease dynamics
- **Reporting delay modeling**: Explicit handling of surveillance system delays
- **Uncertainty propagation**: Full posterior distributions for risk assessment
- **Transform robustness**: Appropriate handling of rates, proportions, and counts

## Performance for Operational Surveillance

- **Real-time capability**: Optimized for operational surveillance timelines
- **Scalable architecture**: Handles multiple surveillance streams simultaneously
- **Memory efficiency**: Designed for continuous operational use
- **Parallel ensemble generation**: Accelerated uncertainty quantification

## CDC Development Standards

This package adheres to:

- **CDC software development** guidelines and best practices
- **Federal cybersecurity** requirements and protocols
- **Scientific reproducibility** standards for epidemiological research
- **Open government** principles for transparency and collaboration

## API Reference and Documentation

- **[Complete API Reference](api.md)**: Detailed function documentation with epidemiological examples
- **User Guides**: Step-by-step tutorials for common surveillance scenarios
- **Methodology Documentation**: Mathematical background and validation studies
- **Case Studies**: Real-world applications in public health surveillance

## Support and Collaboration

### For CDC Staff and Partners
- **Internal collaboration**: Contact CFA/OED for institutional support
- **Training resources**: Available through CDC learning management systems
- **Technical support**: Dedicated support for operational implementations

### For External Researchers
- **GitHub Issues**: Technical questions and bug reports
- **Scientific collaboration**: Research partnerships and method development
- **Community engagement**: Contributing to epidemiological modeling community

## Compliance and Quality Assurance

NowcastAutoGP.jl meets requirements for:

- **508 Accessibility**: Documentation and interfaces accessible to all users
- **Data privacy**: No personally identifiable information in package operations
- **Scientific integrity**: Peer-reviewed methodology and transparent validation
- **Government standards**: Compliance with federal software development guidelines

---

## Citation and Attribution

When using NowcastAutoGP.jl in research, surveillance reports, or policy documents:

```bibtex
@software{nowcastautogp2024,
  title={NowcastAutoGP.jl: Gaussian Process Nowcasting for Public Health Surveillance},
  author={Centers for Disease Control and Prevention, Center for Forecasting and Outbreak Analytics},
  year={2024},
  url={https://github.com/CDCgov/NowcastAutoGP.jl},
  note={U.S. Government Work - Public Domain}
}
```

## Legal and Licensing

This repository constitutes a **work of the United States Government** and is not subject to domestic copyright protection under 17 USC § 105. This repository is in the **public domain** within the United States, and copyright and related rights in the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).

**General disclaimer**: This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm). GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software.

---

*Developed by the Centers for Disease Control and Prevention*
*Saving Lives, Protecting People • CDC.gov*
