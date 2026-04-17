#include <string>

#include "AnalyticCoordinateGrid.hpp"
#include "AthenaFieldReader.hpp"
#include "Field.hpp"
#include "FieldBinaryIO.hpp"
#include "FieldBinaryWriter.hpp"
#include "FieldInterpolator.hpp"
#include "FieldOperator.hpp"
#include "FieldOutputCommon.hpp"
#include "FieldOutputTypes.hpp"
#include "FocusCoefficient.hpp"
#include "FocusDebuger.hpp"
#include "FocusSolver.hpp"
#include "Grid.hpp"
#include "GridValidation.hpp"
#include "KokkosDevice.hpp"
#include "LegencyModel.hpp"
#include "MappedCoordinateGrid.hpp"
#include "ParkerCoefficient.hpp"
#include "ParkerDebuger.hpp"
#include "ParkerSolver.hpp"
#include "ParticleGenerator.hpp"
#include "ParticleSystem.hpp"
#include "ParticleSystemDebugger.hpp"
#include "PhysicalConstant.hpp"
#include "QLTModel.hpp"
#include "RandomManager.hpp"
#include "SolverBase.hpp"
#include "StoredCoordinateGrid.hpp"
#include "TurbulenceProperties.hpp"
#include "UnitConverter.hpp"

namespace {

[[maybe_unused]] void instantiate_athena_reader_for_compile_smoke(
    const std::string& path) {
    const AthenaFieldIO::AthenaBinaryFieldReadOptions<> options;
    const auto background =
        AthenaFieldIO::read_athena_binary_background_field(path, options);
    (void)background.nx;
}

} // namespace
