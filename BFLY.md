# TF2 and TFLite
A this fork contains custom TF2/TFLite code needed
by the DL team for iOS model support. See the TF2 online
documentation for bazel related targets and build info.

In particular have some TFLite kernel code ported from
the tf-addons repo here and some op registration
for stateful variables.

The principal iOS target here would be

    //tensorflo/lite/ios:TensorFlowLiteC_framework

## History
**Note**: branch `master` has moved to a tf2;
 tag v2.5.0_rc2 as of this note, but overtime it will evolve
 as new releases are made.

The old TF1 `master` branch, in production as of early 2021Q2 has
been moved to the branch `tf1`.  This legacy branch, like
`master`, has been
setup as a protected branch in github, and requires PR
review prior to submission.

