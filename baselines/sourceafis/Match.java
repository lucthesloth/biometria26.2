import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintImageOptions;
import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintTemplate;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Scores every pair of a pairs file with SourceAFIS. Args: manifest.csv (sample,path,dpi) pairs.csv out.csv */
public class Match {
    public static void main(String[] args) throws Exception {
        Map<String, FingerprintTemplate> templates = new HashMap<>();
        List<String> manifest = Files.readAllLines(Path.of(args[0]), StandardCharsets.UTF_8);
        for (String line : manifest.subList(1, manifest.size())) {
            String[] f = line.split(",");
            var image = new FingerprintImage(Files.readAllBytes(Path.of(f[1])),
                    new FingerprintImageOptions().dpi(Double.parseDouble(f[2])));
            templates.put(f[0], new FingerprintTemplate(image));
        }
        List<String> pairs = Files.readAllLines(Path.of(args[1]), StandardCharsets.UTF_8);
        Map<String, FingerprintMatcher> matchers = new HashMap<>();
        try (PrintWriter out = new PrintWriter(args[2], StandardCharsets.UTF_8)) {
            out.println("amostra_1,amostra_2,tipo,score");
            for (String line : pairs.subList(1, pairs.size())) {
                String[] f = line.split(",");
                var matcher = matchers.computeIfAbsent(f[0], k -> new FingerprintMatcher(templates.get(k)));
                out.println(f[0] + "," + f[1] + "," + f[2] + "," + matcher.match(templates.get(f[1])));
            }
        }
    }
}
