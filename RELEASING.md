- [Overview](#overview)
- [Branching](#branching)
  - [Single Repo Artifacts Branching](#single-repo-artifacts-branching)
  - [Feature Branches](#feature-branches)
- [Versioning](#versioning)
  - [Version Numbers](#version-numbers)
    - [Remote-Vector-Index-Build-Service Version Numbers](#remote-vector-index-build-service-version-numbers)
- [Tagging](#tagging)
- [Release Labels](#release-labels)

## Overview

This document explains the release strategy for artifacts in the remote-vector-index-service-worker.

## Branching

Projects create a new branch when they need to start working on 2 separate versions of the product, with the `main` branch being the furthermost release. 

### Single Repo Artifacts Branching

For the initial first release, remote-vector-index-build-service follows a simpler branching model with the next release always living on `main` and no patch branches.

### Feature Branches

Do not creating branches in the upstream repo, use your fork, for the exception of long lasting feature branches that require active collaboration from multiple developers. Name feature branches `feature/<thing>`. Once the work is merged to `main`, please make sure to delete the feature branch.

## Versioning

All distributions in this organization [follow SemVer](https://opensearch.org/blog/what-is-semver/). A user-facing breaking change can only be made in a major release. Any regression that breaks SemVer is considered a high severity bug.

### Version Numbers

#### Remote-Vector-Index-Build-Service Version Numbers

The build number of the service is 3-digit `major.minor.patch` (e.g. `1.9.0`)

### Tagging

Create tags after a release that match the version number, `major.minor.patch`, without a `v` prefix.

### Release Labels

Repositories create consistent release labels, such as `v1.0.0`, `v1.1.0` and `v2.0.0`. Use release labels to target an issue or a PR for a given release. See [MAINTAINERS](MAINTAINERS.md#triage-open-issues) for more information on triaging issues.