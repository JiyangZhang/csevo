package org.csevo;

import com.github.javaparser.ParseProblemException;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.google.gson.stream.JsonWriter;
import data.MethodData;
import data.MethodProjectRevision;
import data.ProjectData;
import org.csevo.util.BashUtils;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import static org.csevo.Collector.GSON;
import static org.csevo.Collector.GSON_NO_PPRINT;
import static org.csevo.Collector.log;
import static org.csevo.Collector.sConfig;

public class MethodDataCollector {
    
    private static ProjectData sProjectData;
    
    private static Map<Integer, Integer> sMethodDataIdHashMap = new HashMap<>();
    private static int sCurrentMethodDataId = 0;
    private static List<MethodProjectRevision> sMethodProjectRevisionList = new LinkedList<>();
    private static Map<Integer, List<Integer>> sFileCache = new HashMap<>();
    
    private static JsonWriter sMethodDataWriter;
    private static JsonWriter sMethodProjectRevisionWriter;
    
    public static void collect() {
        try {
            // 1. Load project data
            sProjectData = GSON.fromJson(new FileReader(Paths.get(sConfig.projectDataFile).toFile()), ProjectData.class);
            log("Processing " + sProjectData.name +", got " + sProjectData.revisions.size() + " revisions to process.");
            
            // 2. Init the writers for saving
            sMethodDataWriter = GSON.newJsonWriter(new FileWriter(sConfig.outputDir + "/method-data.json"));
            sMethodDataWriter.beginArray();
            sMethodProjectRevisionWriter = GSON_NO_PPRINT.newJsonWriter(new FileWriter(sConfig.outputDir + "/method-project-revision.json"));
            sMethodProjectRevisionWriter.beginArray();

            if (sConfig.year) {
                for (Map.Entry<String, String> yealRevision: sProjectData.yearRevisions.entrySet()) {
                    log("Revision on " + yealRevision.getKey());
                    collectRevision(yealRevision.getValue(), yealRevision.getKey());
                }
            }
            else {
                // 3. Process each revision
                int i = 0;
                for (String revision : sProjectData.revisions) {
                    ++i;
                    log("Revision " + i + "/" + sProjectData.revisions.size());
                    collectRevision(revision);
                }
            }
            
            // -1. Close readers
            sMethodDataWriter.endArray();
            sMethodDataWriter.close();
            sMethodProjectRevisionWriter.endArray();
            sMethodProjectRevisionWriter.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void collectRevision(String revision, String year) throws IOException {
        // 1. Check out the revision
        if (revision.isEmpty()) {
            // Create and save MethodProjectRevision
            MethodProjectRevision methodProjectRevision = new MethodProjectRevision();
            methodProjectRevision.prjName = sProjectData.name;
            methodProjectRevision.revision = revision;
            methodProjectRevision.year = year;
            methodProjectRevision.methodIds = new LinkedList<>(); ;
            addMethodProjectRevision(methodProjectRevision);

            log("Parsed 0 files. " +
                    "Reused 0 files. " +
                    "Parsing error for 0 files. " +
                    "Ignored 0 methods. " +
                    "Total collected " + sMethodDataIdHashMap.size() + " methods.");
            return;
        }
        else {
            BashUtils.run("cd " + sConfig.projectDir + " && git checkout -f " + revision, 0);
        }
        // 2. Find all java files
        Path projectPath = Paths.get(sConfig.projectDir);
        List<Path> javaFiles = Files.walk(projectPath)
                .filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".java"))
                .sorted(Comparator.comparing(Object::toString))
                .collect(Collectors.toList());
        // For openjdk_jdk, ignore test files because many of them are stress tests for Java parsers
//        if (sProjectData.name.equals("openjdk_jdk") || sProjectData.name.equals("openjdk_loom") || sProjectData.name.equals("antlr_antlr4")) {
//            javaFiles.removeIf(p -> projectPath.relativize(p).toString().contains("test"));
//        }
        // ignore tests
        javaFiles.removeIf(p -> projectPath.relativize(p).toString().contains("test"));
        log("In revision " + revision + ", got " + javaFiles.size() + " files to parse");

        // 3. For each java file, parse and get methods
        MethodDataCollectorVisitor visitor = new MethodDataCollectorVisitor();
        List<Integer> idsRevision = new LinkedList<>();
        int parseErrorCount = 0;
        int ignoredCount = 0;
        int reuseFileCount = 0;
        int parseFileCount = 0;
        for (Path javaFile : javaFiles) {
            // Skip parsing identical files, just add the ids
            int fileHash = getFileHash(javaFile);
            List<Integer> idsFile = sFileCache.get(fileHash);

            if (idsFile == null) {
                // Actually parse this file and collect ids
                idsFile = new LinkedList<>();
                String path = projectPath.relativize(javaFile).toString();

                MethodDataCollectorVisitor.Context context = new MethodDataCollectorVisitor.Context();
                try {
                    CompilationUnit cu = StaticJavaParser.parse(javaFile);
                    cu.accept(visitor, context);
                } catch (ParseProblemException e) {
                    ++parseErrorCount;
                }

                ignoredCount += context.ignoredCount;

                for (MethodData methodData : context.methodDataList) {
                    // Reuse (for duplicate data) or allocate the data id
                    methodData.path = path;
                    int methodId = addMethodData(methodData);
                    idsFile.add(methodId);
                }

                // Update file cache
                sFileCache.put(fileHash, idsFile);
                ++parseFileCount;
            } else {
                ++reuseFileCount;
            }

            idsRevision.addAll(idsFile);
        }

        // Create and save MethodProjectRevision
        MethodProjectRevision methodProjectRevision = new MethodProjectRevision();
        methodProjectRevision.prjName = sProjectData.name;
        methodProjectRevision.revision = revision;
        methodProjectRevision.year = year;
        methodProjectRevision.methodIds = idsRevision;
        addMethodProjectRevision(methodProjectRevision);

        log("Parsed " + parseFileCount + " files. " +
                "Reused " + reuseFileCount + " files. " +
                "Parsing error for " + parseErrorCount + " files. " +
                "Ignored " + ignoredCount + " methods. " +
                "Total collected " + sMethodDataIdHashMap.size() + " methods.");
    }

    private static void collectRevision(String revision) throws IOException {
        // 1. Check out the revision
        BashUtils.run("cd " + sConfig.projectDir + " && git checkout -f " + revision, 0);

        // 2. Find all java files
        Path projectPath = Paths.get(sConfig.projectDir);
        List<Path> javaFiles = Files.walk(projectPath)
                .filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".java"))
                .sorted(Comparator.comparing(Object::toString))
                .collect(Collectors.toList());
        // For openjdk_jdk, ignore test files because many of them are stress tests for Java parsers
        if (sProjectData.name.equals("openjdk_jdk")) {
            javaFiles.removeIf(p -> projectPath.relativize(p).toString().contains("test"));
        }
        log("In revision " + revision +", got " + javaFiles.size() + " files to parse");

        // 3. For each java file, parse and get methods
        MethodDataCollectorVisitor visitor = new MethodDataCollectorVisitor();
        List<Integer> idsRevision = new LinkedList<>();
        int parseErrorCount = 0;
        int ignoredCount = 0;
        int reuseFileCount = 0;
        int parseFileCount = 0;
        for (Path javaFile : javaFiles) {
            // Skip parsing identical files, just add the ids
            int fileHash = getFileHash(javaFile);
            List<Integer> idsFile = sFileCache.get(fileHash);

            if (idsFile == null) {
                // Actually parse this file and collect ids
                idsFile = new LinkedList<>();
                String path = projectPath.relativize(javaFile).toString();

                MethodDataCollectorVisitor.Context context = new MethodDataCollectorVisitor.Context();
                try {
                    CompilationUnit cu = StaticJavaParser.parse(javaFile);
                    cu.accept(visitor, context);
                } catch (ParseProblemException e) {
                    ++parseErrorCount;
                }

                ignoredCount += context.ignoredCount;

                for (MethodData methodData : context.methodDataList) {
                    // Reuse (for duplicate data) or allocate the data id
                    methodData.path = path;
                    int methodId = addMethodData(methodData);
                    idsFile.add(methodId);
                }

                // Update file cache
                sFileCache.put(fileHash, idsFile);
                ++parseFileCount;
            } else {
                ++reuseFileCount;
            }

            idsRevision.addAll(idsFile);
        }

        // Create and save MethodProjectRevision
        MethodProjectRevision methodProjectRevision = new MethodProjectRevision();
        methodProjectRevision.prjName = sProjectData.name;
        methodProjectRevision.revision = revision;
        methodProjectRevision.methodIds = idsRevision;
        addMethodProjectRevision(methodProjectRevision);

        log("Parsed " + parseFileCount + " files. " +
                "Reused " + reuseFileCount + " files. " +
                "Parsing error for " + parseErrorCount + " files. " +
                "Ignored " + ignoredCount + " methods. " +
                "Total collected " + sMethodDataIdHashMap.size() + " methods.");
    }


    private static int getFileHash(Path javaFile) throws IOException {
        // Hash both the path and the content
        return Objects.hash(javaFile.toString(), Arrays.hashCode(Files.readAllBytes(javaFile)));
    }
    
    private static int addMethodData(MethodData methodData) {
        // Don't duplicate previous appeared methods (keys: path, code, comment)
        int hash = Objects.hash(methodData.path, methodData.code, methodData.comment);
        Integer prevMethodDataId = sMethodDataIdHashMap.get(hash);
        if (prevMethodDataId != null) {
            // If this method data already existed before, retrieve its id
            return prevMethodDataId;
        } else {
            // Allocate a new id and save this data to the hash map
            methodData.id = sCurrentMethodDataId;
            methodData.prjName = sProjectData.name;
            ++sCurrentMethodDataId;
            sMethodDataIdHashMap.put(hash, methodData.id);
            
            // Save the method data
            GSON.toJson(methodData, MethodData.class, sMethodDataWriter);
            return methodData.id;
        }
    }
    
    private static void addMethodProjectRevision(MethodProjectRevision methodProjectRevision) {
        // Directly write to file
        GSON_NO_PPRINT.toJson(methodProjectRevision, MethodProjectRevision.class, sMethodProjectRevisionWriter);
    }
}
